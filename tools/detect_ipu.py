import json
from collections import defaultdict
from typing import List, Dict, Any

SILENCE_THRESHOLD = 0.2  # 無音と判定する閾値


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


def calculate_pause_duration(
    ipus_by_speaker: Dict[str, List[List[Dict[str, Any]]]],
) -> Dict[str, float]:
    pause_by_speaker = {}

    for speaker, ipus in ipus_by_speaker.items():
        total_pause = 0.0
        for i in range(1, len(ipus)):
            prev_end = ipus[i - 1][-1]["end"]
            curr_start = ipus[i][0]["start"]
            pause = curr_start - prev_end
            if pause >= SILENCE_THRESHOLD:
                total_pause += pause
        pause_by_speaker[speaker] = total_pause

    return pause_by_speaker


# ========== 使用例 ==========

with open(
    "/home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/CSJ/text/D01F0002.json",
    encoding="utf-8",
) as f:
    data = json.load(f)

ipus = group_ipus_by_speaker(data)
pause_durations = calculate_pause_duration(ipus)

# 出力
for speaker, pause in pause_durations.items():
    print(f"話者 {speaker} の Pause 合計時間: {pause:.3f} 秒")
