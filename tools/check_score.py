import json, statistics

path = "data_tabidachi/moshi_stage3_new_jchat_tabidachi/evaluated_dialog_propmt_changed.jsonl"  # ←ファイル名に合わせて変更
scores = []

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        s = obj.get("score")
        if isinstance(s, (int, float)):  # Noneや文字列は弾く
            scores.append(s)

if scores:
    mean = statistics.mean(scores)
    print(f"count={len(scores)}, mean={mean:.4f}, min={min(scores)}, max={max(scores)}")
else:
    print("有効な score が見つかりませんでした。")
