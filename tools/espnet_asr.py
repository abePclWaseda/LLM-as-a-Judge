# tools/espnet_asr.py
import argparse
import glob
import json
import os
import sys
from typing import List

import soundfile as sf
from reazonspeech.espnet.asr import (
    load_model,
    transcribe,
    audio_from_numpy,
)


def list_wavs(patterns: List[str]) -> List[str]:
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    # 重複排除 & 安定順序
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe stereo wavs with ESPnet and output JSONL (dialog view)."
    )
    parser.add_argument(
        "--wav-glob",
        nargs="+",
        required=True,
        help='One or more glob patterns for wav files, e.g. "/path/*.wav" "/path2/*.wav"',
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for ESPnet model (default: cuda).",
    )
    parser.add_argument(
        "--skip-non-stereo",
        action="store_true",
        help="Skip files that are not stereo (default: True).",
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if exists (default: False).",
    )

    args = parser.parse_args()

    wav_files = list_wavs(args.wav_glob)
    if not wav_files:
        print("No wav files matched the given patterns.", file=sys.stderr)
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(args.output) and not args.overwrite:
        print(
            f"Output already exists: {args.output} (use --overwrite to replace)",
            file=sys.stderr,
        )
        return 2

    print(f"[Info] Loading ESPnet model on device={args.device}", flush=True)
    model = load_model(device=args.device)

    written = 0
    skipped = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for wav in wav_files:
            try:
                waveform, sr = sf.read(wav)
            except Exception as e:
                print(f"[Warn] Failed to read {wav}: {e}", file=sys.stderr, flush=True)
                skipped += 1
                continue

            if waveform.ndim != 2 or waveform.shape[1] != 2:
                if args.skip_non_stereo:
                    print(f"[Skip] Not stereo: {wav}", flush=True)
                    skipped += 1
                    continue
                else:
                    print(
                        f"[Warn] Not stereo but processing first channel: {wav}",
                        flush=True,
                    )
                    waveform = waveform.reshape(-1, 1)
                    waveform = (
                        waveform
                        if waveform.shape[1] == 2
                        else (waveform if waveform.ndim == 2 else waveform[:, None])
                    )

            try:
                audio0 = audio_from_numpy(waveform[:, 0], sr)
                audio1 = audio_from_numpy(waveform[:, 1], sr)

                result_0 = transcribe(model, audio0)
                result_1 = transcribe(model, audio1)

                all_segments = []
                for seg in result_0.segments:
                    all_segments.append(
                        {"speaker": "A", "start": seg.start_seconds, "text": seg.text}
                    )
                for seg in result_1.segments:
                    all_segments.append(
                        {"speaker": "B", "start": seg.start_seconds, "text": seg.text}
                    )

                all_segments.sort(key=lambda x: x["start"])
                dialogue_lines = [
                    f'{seg["speaker"]}: {seg["text"]}' for seg in all_segments
                ]
                dialogue_text = "\n".join(dialogue_lines)

                obj = {
                    "audio_path": wav,
                    "channel_0_text": result_0.text,
                    "channel_1_text": result_1.text,
                    "dialogue_text": dialogue_text,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
                if written % 10 == 0:
                    print(f"[Info] Wrote {written} items so far...", flush=True)
            except Exception as e:
                print(
                    f"[Warn] Failed to process {wav}: {e}", file=sys.stderr, flush=True
                )
                skipped += 1
                continue

    print(
        f"[Done] total={len(wav_files)} written={written} skipped={skipped} -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
