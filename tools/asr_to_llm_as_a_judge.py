# ステレオ音声を reazon-espnet-asr で書き起こしをして，それを "llm-jp/llm-jp-3-7.2b" で対話の質を評価します．

import json
import glob
from reazonspeech.espnet.asr import load_model, transcribe, audio_from_path

# モデルロード
model = load_model(device="cuda")

# ステレオ音声ファイル群を読み込む
wav_files = glob.glob(
    "/home/acg17145sv/experiments/0162_dialogue_model/moshi-finetune/output/moshi_stage3_tabidachi/step_498_fp32/continuation_jchat/generated_wavs/*.wav"
)

# 出力ファイル
output_path = "transcripts.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for wav in wav_files:
        audio = audio_from_path(wav)
        ret = transcribe(model, audio)

        obj = {"audio_path": wav, "text": ret.text}
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"書き起こし結果を {output_path} に保存しました。")
