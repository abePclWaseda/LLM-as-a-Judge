import json
from collections import defaultdict
from typing import List, Dict, Any

# 閾値（秒）
SILENCE_THRESHOLD = 0.2


def group_ipus_by_speaker(
    words: List[Dict[str, Any]],
) -> Dict[str, List[List[Dict[str, Any]]]]:
    # 話者ごとに発話リストをまとめる
    speakers = defaultdict(list)
    for word in words:
        speakers[word["speaker"]].append(word)

    # 各話者について IPU 分割
    ipus_by_speaker = {}
    for speaker, word_list in speakers.items():
        # 開始時刻順にソート
        sorted_words = sorted(word_list, key=lambda w: w["start"])

        ipus = []
        current_ipu = [sorted_words[0]]

        for prev, curr in zip(sorted_words, sorted_words[1:]):
            # 無音の長さ
            pause = curr["start"] - prev["end"]
            if pause >= SILENCE_THRESHOLD:
                ipus.append(current_ipu)
                current_ipu = [curr]
            else:
                current_ipu.append(curr)
        ipus.append(current_ipu)

        ipus_by_speaker[speaker] = ipus

    return ipus_by_speaker


# ========== 使用例 ==========

# JSONデータの読み込み
with open(
    "/home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/CSJ/text/D01F0002.json",
    encoding="utf-8",
) as f:
    data = json.load(f)

# IPU 分割実行
result = group_ipus_by_speaker(data)

# 出力（整形）
for speaker, ipus in result.items():
    print(f"\n話者 {speaker}: {len(ipus)} IPUs")
    for i, ipu in enumerate(ipus, 1):
        text = "".join(word["word"] for word in ipu)
        start = ipu[0]["start"]
        end = ipu[-1]["end"]
        print(f"  IPU{i}: {start:.2f} - {end:.2f} 秒 | '{text}'")
