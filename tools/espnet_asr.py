import json
import glob
import soundfile as sf
from reazonspeech.espnet.asr import (
    load_model,
    transcribe,
    audio_from_numpy,
)  # ← audio_from_numpy を使う

# モデルロード
model = load_model(device="cuda")

# ステレオ音声ファイル群を読み込む
wav_files = glob.glob(
    "/home/acg17145sv/experiments/0162_dialogue_model/moshi-finetune/output/moshi_stage3_old_jchat_clean_tabidachi/step_498_fp32/continuation_jchat/generated_wavs/*.wav"
)

# 出力ファイル
output_path = "transcripts.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for wav in wav_files:
        waveform, sr = sf.read(wav)
        if waveform.ndim != 2 or waveform.shape[1] != 2:
            print(f"スキップ（ステレオではない）: {wav}")
            continue

        # 各チャネルを AudioData に変換（numpy → AudioData）
        audio0 = audio_from_numpy(waveform[:, 0], sr)
        audio1 = audio_from_numpy(waveform[:, 1], sr)

        # 書き起こし
        result_0 = transcribe(model, audio0)
        result_1 = transcribe(model, audio1)

        # Segmentに話者ラベルを付けて統合
        all_segments = []
        for seg in result_0.segments:
            all_segments.append(
                {"speaker": "A", "start": seg.start_seconds, "text": seg.text}
            )
        for seg in result_1.segments:
            all_segments.append(
                {"speaker": "B", "start": seg.start_seconds, "text": seg.text}
            )

        # 開始時刻でソート
        all_segments.sort(key=lambda x: x["start"])

        # 会話テキストの構築
        dialogue_lines = [f'{seg["speaker"]}: {seg["text"]}' for seg in all_segments]
        dialogue_text = "\n".join(dialogue_lines)

        # JSONLとして出力
        obj = {
            "audio_path": wav,
            "channel_0_text": result_0.text,
            "channel_1_text": result_1.text,
            "dialogue_text": dialogue_text,
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"時系列対話を書き起こした結果を {output_path} に保存しました。")
