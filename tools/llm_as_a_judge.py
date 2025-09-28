import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 使用モデル（13B instruct4, chat形式） ─────────────────────
MODEL_NAME = "llm-jp/llm-jp-3.1-13b-instruct4"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# pad_token 未設定対策（必要なら）
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def evaluate_text(text: str) -> dict:
    # 4観点を考慮しつつ、最終出力は 1〜10 の整数ひとつ
    system_msg = (
        "以下は、タスクを説明する指示です。要求を正確に満たす応答を書きなさい。"
    )
    user_msg = f"""
次の対話テキストを、以下の4観点を総合的に考慮して評価してください。
- 一貫性 (Coherence)
- 自然さ (Fluency & Naturalness)
- 関連性 (Relevance)
- 情報量 (Informativeness)

出力は、1〜10の**整数**で表す**総合スコア**を**ひとつだけ**にしてください。
数字以外の文字は一切出力しないでください（例: 7）。

対話テキスト:
\"\"\"{text}\"\"\"""".strip()

    chat = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    inputs = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=8,  # 数字のみ想定で短め
            do_sample=False,  # 評価用途なので決定的に
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

    # 生成部分のみを取り出してデコード
    gen_only = output[inputs.shape[-1] :]
    answer = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

    # 1〜10 の整数のみ抽出
    m = re.search(r"\b(10|[1-9])\b", answer)
    if m:
        return {"score": int(m.group(1))}
    return {"score": None}


# ── 入出力 ───────────────────────────────────────────────────
input_path = "transcripts.jsonl"
output_path = "evaluated.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(
    output_path, "w", encoding="utf-8"
) as fout:
    for line in fin:
        obj = json.loads(line)
        scores = evaluate_text(obj["text"])
        obj.update(scores)  # {"score": <int or None>}
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"評価結果を {output_path} に保存しました。")
