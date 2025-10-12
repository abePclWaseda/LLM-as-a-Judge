import json
from collections import defaultdict
from typing import List, Dict, Any
import math

SILENCE_THRESHOLD = 0.2  # 無音判定の閾値（秒）


def group_ipus_by_speaker(
    words: List[Dict[str, Any]],
) -> Dict[str, List[List[Dict[str, Any]]]]:
    speakers = defaultdict(list)
    for word in words:
        speakers[word["speaker"]].append(word)

    ipus_by_speaker = {}
    for speaker, word_list in speakers.items():
        sorted_words = sorted(word_list, key=lambda w: w["start"])
        ipus = []
        current_ipu = [sorted_words[0]]

        for prev, curr in zip(sorted_words, sorted_words[1:]):
            pause = curr["start"] - prev["end"]
            if pause >= SILENCE_THRESHOLD:
                ipus.append(current_ipu)
                current_ipu = [curr]
            else:
                current_ipu.append(curr)
        ipus.append(current_ipu)
        ipus_by_speaker[speaker] = ipus

    return ipus_by_speaker


def count_ipus_per_minute(
    ipus_by_speaker: Dict[str, List[List[Dict[str, Any]]]],
) -> Dict[str, Dict[int, int]]:
    counts = {}
    for speaker, ipus in ipus_by_speaker.items():
        minute_count = defaultdict(int)
        for ipu in ipus:
            start_time = ipu[0]["start"]
            minute = int(start_time // 60)  # 何分目か
            minute_count[minute] += 1
        counts[speaker] = dict(sorted(minute_count.items()))
    return counts


def get_ipu_durations(
    ipus_by_speaker: Dict[str, List[List[Dict[str, Any]]]],
) -> Dict[str, List[float]]:
    durations = {}
    for speaker, ipus in ipus_by_speaker.items():
        durations[speaker] = [
            round(ipu[-1]["end"] - ipu[0]["start"], 3) for ipu in ipus
        ]
    return durations


# ========== 実行 ==========

with open("/home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/CSJ/text/D01F0002.json", encoding="utf-8") as f:
    data = json.load(f)

ipus = group_ipus_by_speaker(data)

# 1分ごとのIPU数
ipu_counts = count_ipus_per_minute(ipus)

# IPUの継続時間
ipu_durations = get_ipu_durations(ipus)

# 出力
print("📊 IPU数（1分ごと）:")
for speaker, minute_counts in ipu_counts.items():
    print(f"話者 {speaker}:")
    for minute, count in minute_counts.items():
        print(f"  {minute}分目: {count} IPUs")

print("\n⏱️ IPU継続時間（秒）:")
for speaker, durations in ipu_durations.items():
    print(f"話者 {speaker}:")
    print("  " + ", ".join(f"{d:.3f}" for d in durations))
