#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from reazonspeech.espnet.asr import (
    load_model,
    transcribe,
    audio_from_numpy,
)

# ===== utils =====


def list_wavs(patterns: List[str]) -> List[str]:
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))


@dataclass(frozen=True)
class Seg:
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _get_seg_times(seg) -> Tuple[Optional[float], Optional[float], str]:
    """ReazonSpeech/ESPnet の seg から (start,end,text) を最善努力で取得。"""
    # start
    s = getattr(seg, "start_seconds", None)
    if s is None:
        s = getattr(seg, "start", None)
    try:
        s = float(s) if s is not None else None
    except Exception:
        s = None

    # end
    e = getattr(seg, "end_seconds", None)
    if e is None:
        d = getattr(seg, "duration_seconds", None)
        if d is None:
            d = getattr(seg, "duration", None)
        if d is not None and s is not None:
            try:
                e = float(s) + float(d)
            except Exception:
                e = None
    else:
        try:
            e = float(e)
        except Exception:
            e = None

    txt = getattr(seg, "text", "") or ""
    return s, e, str(txt)


def _close_open_ends(spans: List[Seg]) -> List[Seg]:
    """
    end が無い/おかしいセグメントは「次セグメントの start まで」で補完。
    それも無ければ 0.04s を仮置き（短すぎるので後段で落ちやすい）。
    """
    if not spans:
        return spans
    spans = sorted(spans, key=lambda z: (z.start, z.end))
    fixed = []
    for i, z in enumerate(spans):
        if (z.end is None) or (z.end <= z.start):
            nxt = spans[i + 1].start if i + 1 < len(spans) else None
            if nxt is not None and nxt > z.start:
                fixed.append(Seg(z.start, nxt, z.text))
            else:
                fixed.append(Seg(z.start, z.start + 0.04, z.text))
        else:
            fixed.append(z)
    return fixed


def _merge_short_gaps(segs: List[Seg], min_silence: float) -> List[Seg]:
    """隣接ギャップ < min_silence なら結合（text も連結）。"""
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: (s.start, s.end))
    out = [segs[0]]
    for s in segs[1:]:
        prev = out[-1]
        gap = s.start - prev.end
        if gap < min_silence:
            out[-1] = Seg(
                prev.start, max(prev.end, s.end), (prev.text + " " + s.text).strip()
            )
        else:
            out.append(s)
    return out


def _filter_min_len(segs: List[Seg], min_ipu_sec: float) -> List[Seg]:
    return [z for z in segs if (z.end > z.start) and (z.duration >= min_ipu_sec)]


def _segments_to_ipus(
    espnet_segments, min_silence: float, min_ipu_sec: float, merge_short_gaps: bool
) -> List[Seg]:
    """
    ESPnet segments -> IPUs
      - merge_short_gaps=False: segmentsをそのまま使う（start/endの補完とmin長フィルタのみ）
      - merge_short_gaps=True : 短いギャップは結合してIPU化
    """
    spans = []
    for seg in espnet_segments or []:
        s, e, txt = _get_seg_times(seg)
        if s is None:
            continue
        if e is None:
            # end欠損は一旦 start と同じにして後段で補完
            spans.append(Seg(float(s), float(s), txt))
        else:
            spans.append(Seg(float(s), float(e), txt))

    # end の欠損/不整合を補完
    spans = _close_open_ends(spans)

    if merge_short_gaps:
        ipus = _merge_short_gaps(spans, min_silence=min_silence)
    else:
        # マージしない: segmentsをそのまま（補完済み）IPUとみなす
        ipus = sorted(spans, key=lambda s: (s.start, s.end))

    # 短すぎるIPUは除外
    ipus = _filter_min_len(ipus, min_ipu_sec=min_ipu_sec)
    return ipus


# ===== main =====


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build IPUs from ReazonSpeech ESPnet timestamps and output JSONL (stereo=two speakers)."
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
        "--min_silence",
        type=float,
        default=0.2,
        help="Gap >= this splits IPU (sec).",
    )
    parser.add_argument(
        "--min_ipu_sec",
        type=float,
        default=0.15,
        help="Drop IPU shorter than this (sec).",
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
    parser.add_argument(
        "--merge-short-gaps",
        action="store_true",
        help="If set, merge adjacent segments when gap < --min_silence. Default: do NOT merge (use raw ESPnet segments as IPUs).",
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
                    waveform = np.atleast_2d(waveform)
                    if waveform.shape[0] < waveform.shape[1]:
                        waveform = waveform.T
                    if waveform.shape[1] == 1:
                        waveform = np.concatenate([waveform, waveform], axis=1)

            try:
                audio0 = audio_from_numpy(waveform[:, 0], sr)
                audio1 = audio_from_numpy(waveform[:, 1], sr)

                result_0 = transcribe(model, audio0)  # channel A
                result_1 = transcribe(model, audio1)  # channel B

                ipus_A = _segments_to_ipus(
                    getattr(result_0, "segments", None),
                    args.min_silence,
                    args.min_ipu_sec,
                    merge_short_gaps=args.merge_short_gaps,
                )
                ipus_B = _segments_to_ipus(
                    getattr(result_1, "segments", None),
                    args.min_silence,
                    args.min_ipu_sec,
                    merge_short_gaps=args.merge_short_gaps,
                )
                # サマリ
                sum_A = float(sum(z.duration for z in ipus_A))
                sum_B = float(sum(z.duration for z in ipus_B))

                # JSONL 1行
                obj = {
                    "audio_path": wav,
                    "sample_rate": sr,
                    "duration_sec": round(len(waveform) / sr, 3),
                    "speakers": ["A", "B"],
                    "channel_0_ipus": [
                        {
                            "start": round(z.start, 3),
                            "end": round(z.end, 3),
                            "duration": round(z.duration, 3),
                            "text": z.text,
                        }
                        for z in ipus_A
                    ],
                    "channel_1_ipus": [
                        {
                            "start": round(z.start, 3),
                            "end": round(z.end, 3),
                            "duration": round(z.duration, 3),
                            "text": z.text,
                        }
                        for z in ipus_B
                    ],
                    "summary": {
                        "total_ipus": len(ipus_A) + len(ipus_B),
                        "A": {"ipus": len(ipus_A), "speaking_sec": round(sum_A, 3)},
                        "B": {"ipus": len(ipus_B), "speaking_sec": round(sum_B, 3)},
                    },
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
