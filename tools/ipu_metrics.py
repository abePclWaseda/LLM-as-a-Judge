#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from typing import List, Tuple, Dict, Any, Optional
from bisect import bisect_left


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def merge_overlap_duration(
    a: List[Tuple[float, float]], b: List[Tuple[float, float]]
) -> float:
    """2 本の区間列の重なり総時間を二重ポインタで計算"""
    a = sorted(a)
    b = sorted(b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        start = max(s1, s2)
        end = min(e1, e2)
        if end > start:
            total += end - start
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return total


def overlap_intervals(
    a: List[Tuple[float, float]], b: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """重なり“区間”も列挙（明細出力用）。"""
    a = sorted(a)
    b = sorted(b)
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        start = max(s1, s2)
        end = min(e1, e2)
        if end > start:
            out.append((start, end))
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return out


def compute_pauses(ipus: List[Tuple[float, float]], min_silence: float, speaker: str):
    """同一話者の連続 IPU 間が min_silence 以上のとき Pause とする。"""
    out = []
    ipus = sorted(ipus)
    for (s0, e0), (s1, e1) in zip(ipus, ipus[1:]):
        gap = max(0.0, s1 - e0)
        if gap >= min_silence:
            out.append({"speaker": speaker, "start": e0, "end": s1, "duration": gap})
    return out


def compute_silences_and_classify(
    A: List[Tuple[float, float]], B: List[Tuple[float, float]], min_silence: float
):
    """
    無音区間（誰も話していない）を Pause / Gap に分類。
      - Pause: 直前の話者 == 直後の話者
      - Gap  : 直前の話者 != 直後の話者
    ここで「直後の話者」は無音終了時刻 t1 そのものに発話開始が
    存在する場合も含めて判定する（t1 を“含む”）。
    """
    events = []
    for s, e in A:
        events.append((s, +1, "A"))
        events.append((e, -1, "A"))
    for s, e in B:
        events.append((s, +1, "B"))
        events.append((e, -1, "B"))
    if not events:
        return [], []

    events.sort()

    # 各 timestamp の「その時刻のイベントをすべて処理し終えた直後」のアクティブ集合
    active_state_by_t: Dict[float, set] = {}
    active = set()
    for t, delta, who in events:
        # 更新前の状態は使わず、「更新後」を保持する
        if delta == +1:
            active.add(who)
        else:
            active.discard(who)
        active_state_by_t[t] = set(active)

    times = sorted(active_state_by_t.keys())

    # t より「前」の時刻群で最後に非空だった集合から話者を特定
    def last_speaker_before(t: float) -> Optional[str]:
        i = bisect_left(times, t)  # t を挿入できる位置（= t 未満の数）
        for k in range(i - 1, -1, -1):
            st = active_state_by_t[times[k]]
            if st:
                return next(iter(st))
        return None

    # t 以降（t を含む）で最初に非空となる集合から話者を特定
    def next_speaker_on_or_after(t: float) -> Optional[str]:
        i = bisect_left(times, t)
        for k in range(i, len(times)):
            st = active_state_by_t[times[k]]
            if st:
                return next(iter(st))
        return None

    pause_list = []
    gap_list = []

    # (t0, t1) の開区間における状態は「t0 直後の状態」に等しい
    # よって t0 の状態が空なら (t0, t1) は無音
    for t0, t1 in zip(times, times[1:]):
        if t1 <= t0:
            continue
        st_after_t0 = active_state_by_t[t0]
        if st_after_t0:
            continue  # 無音ではない

        dur = t1 - t0
        if dur < min_silence:
            continue

        prev_spk = last_speaker_before(t0)
        next_spk = next_speaker_on_or_after(t1)  # ★ t1 を含めて探索

        if prev_spk and next_spk:
            if prev_spk == next_spk:
                pause_list.append(
                    {"speaker": prev_spk, "start": t0, "end": t1, "duration": dur}
                )
            else:
                gap_list.append(
                    {
                        "from_speaker": prev_spk,
                        "to_speaker": next_spk,
                        "start": t0,
                        "end": t1,
                        "duration": dur,
                    }
                )
        # prev または next が無い（冒頭/末尾の無音など）は分類しない

    return pause_list, gap_list


def compute_for_record(rec: Dict[str, Any], min_silence: float):
    # スピーカー名の決定（なければ 'A','B'）
    speakers = rec.get("speakers") or ["A", "B"]
    spkA = speakers[0] if len(speakers) > 0 else "A"
    spkB = speakers[1] if len(speakers) > 1 else "B"

    ch0 = rec.get("channel_0_ipus", [])
    ch1 = rec.get("channel_1_ipus", [])

    A_ipus = sorted(
        [(float(x["start"]), float(x["end"])) for x in ch0], key=lambda t: t[0]
    )
    B_ipus = sorted(
        [(float(x["start"]), float(x["end"])) for x in ch1], key=lambda t: t[0]
    )

    # Pause（同一話者連続IPU）
    pauses_A = compute_pauses(A_ipus, min_silence, spkA)
    pauses_B = compute_pauses(B_ipus, min_silence, spkB)

    # Overlap（重なり総時間 & 区間）
    overlap_sec = merge_overlap_duration(A_ipus, B_ipus)
    overlap_spans = overlap_intervals(A_ipus, B_ipus)

    # 無音区間の分類（Pause/GAP, しきい値適用）
    pauses_sil, gaps = compute_silences_and_classify(A_ipus, B_ipus, min_silence)

    # Pause は2種類の算出方法があるが、通常は「同一話者連続IPU間」を採用
    # → 両者が大きくズレるケース（境界に重なりがある等）だけ注意が必要
    pauses = pauses_A + pauses_B

    return {
        "audio_path": rec.get("audio_path", ""),
        "duration_sec": float(rec.get("duration_sec", 0.0)),
        "ipu_counts": {"A": len(A_ipus), "B": len(B_ipus)},
        "pause_list": pauses,  # 明細（連続IPUベース）
        "pause_total_sec": sum(p["duration"] for p in pauses),
        "gap_list": gaps,  # 明細（無音分類ベース）
        "gap_total_sec": sum(g["duration"] for g in gaps),
        "overlap_total_sec": overlap_sec,
        "overlap_spans": overlap_spans,  # 明細
        "speakers_map": {"A": spkA, "B": spkB},  # 表示用
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compute IPU-based Pause/Gap/Overlap from JSONL."
    )
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to ipu.jsonl (each line is a dict like your examples).",
    )
    ap.add_argument(
        "--min_silence",
        type=float,
        default=0.2,
        help="Silence threshold [sec] to count Pause/Gap (default: 0.2)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory (default: same as input).",
    )
    args = ap.parse_args()

    in_path = args.input
    outdir = args.outdir or os.path.dirname(os.path.abspath(in_path))

    os.makedirs(outdir, exist_ok=True)

    summary_csv = os.path.join(outdir, "ipu_summary.csv")
    pauses_csv = os.path.join(outdir, "ipu_pauses.csv")
    gaps_csv = os.path.join(outdir, "ipu_gaps.csv")
    overlaps_csv = os.path.join(outdir, "ipu_overlaps.csv")

    summaries = []
    pauses_rows = []
    gaps_rows = []
    overlaps_rows = []

    for rec in read_jsonl(in_path):
        R = compute_for_record(rec, args.min_silence)
        bname = os.path.basename(R["audio_path"]) or rec.get("audio_path", "")

        summaries.append(
            {
                "audio_path": bname,
                "dur_sec": f'{R["duration_sec"]:.3f}',
                "IPUs_A": R["ipu_counts"]["A"],
                "IPUs_B": R["ipu_counts"]["B"],
                "Pause_total_sec": f'{R["pause_total_sec"]:.3f}',
                "Gap_total_sec": f'{R["gap_total_sec"]:.3f}',
                "Overlap_total_sec": f'{R["overlap_total_sec"]:.3f}',
            }
        )

        spkmap = R["speakers_map"]
        for p in R["pause_list"]:
            pauses_rows.append(
                {
                    "audio_path": bname,
                    "speaker": (
                        spkmap.get("A")
                        if p["speaker"] == "A"
                        else spkmap.get("B", p["speaker"])
                    ),
                    "start": f'{p["start"]:.3f}',
                    "end": f'{p["end"]:.3f}',
                    "duration": f'{p["duration"]:.3f}',
                }
            )

        for g in R["gap_list"]:
            gaps_rows.append(
                {
                    "audio_path": bname,
                    "from_speaker": spkmap.get(g["from_speaker"], g["from_speaker"]),
                    "to_speaker": spkmap.get(g["to_speaker"], g["to_speaker"]),
                    "start": f'{g["start"]:.3f}',
                    "end": f'{g["end"]:.3f}',
                    "duration": f'{g["duration"]:.3f}',
                }
            )

        for s, e in R["overlap_spans"]:
            overlaps_rows.append(
                {
                    "audio_path": bname,
                    "start": f"{s:.3f}",
                    "end": f"{e:.3f}",
                    "duration": f"{(e - s):.3f}",
                }
            )

    # 書き出し
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=(
                list(summaries[0].keys())
                if summaries
                else [
                    "audio_path",
                    "dur_sec",
                    "IPUs_A",
                    "IPUs_B",
                    "Pause_total_sec",
                    "Gap_total_sec",
                    "Overlap_total_sec",
                ]
            ),
        )
        w.writeheader()
        for row in summaries:
            w.writerow(row)

    with open(pauses_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["audio_path", "speaker", "start", "end", "duration"]
        )
        w.writeheader()
        for row in pauses_rows:
            w.writerow(row)

    with open(gaps_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "audio_path",
                "from_speaker",
                "to_speaker",
                "start",
                "end",
                "duration",
            ],
        )
        w.writeheader()
        for row in gaps_rows:
            w.writerow(row)

    with open(overlaps_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audio_path", "start", "end", "duration"])
        w.writeheader()
        for row in overlaps_rows:
            w.writerow(row)

    print(f"[OK] summary -> {summary_csv}")
    print(f"[OK] pauses  -> {pauses_csv}")
    print(f"[OK] gaps    -> {gaps_csv}")
    print(f"[OK] overlaps-> {overlaps_csv}")


if __name__ == "__main__":
    main()
