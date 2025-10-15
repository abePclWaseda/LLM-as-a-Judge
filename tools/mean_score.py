#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import argparse
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="LLM評価結果のJSONLファイル（judge_raw含む）"
    )
    args = parser.parse_args()

    keys = [
        "coherence",
        "naturalness",
        "relevance",
        "instruction_following",
        "turn_taking",
        "overall",
    ]
    scores = {k: [] for k in keys}

    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                raw = obj.get("judge_raw")
                if not raw:
                    continue
                j = json.loads(raw)
            except Exception:
                continue

            for k in keys:
                v = j.get(k)
                if isinstance(v, (int, float)):
                    scores[k].append(float(v))

    # 各指標の平均と全体平均
    print("=== 平均スコア（1〜10） ===")
    for k in keys:
        if scores[k]:
            avg = statistics.mean(scores[k])
            print(f"{k:22s}: {avg:.2f} ({len(scores[k])}件)")
        else:
            print(f"{k:22s}: -")

    all_vals = [v for k in keys for v in scores[k]]
    if all_vals:
        print("-------------------------")
        print(f"全体平均 (6指標平均): {statistics.mean(all_vals):.2f}")
    else:
        print("スコアがありません。")


if __name__ == "__main__":
    main()
