import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 使用モデル =====
MODEL_NAME = "llm-jp/llm-jp-3.1-13b-instruct4"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

torch.manual_seed(0)  # 決定的に


def evaluate_text(dialogue_text: str) -> dict:
    """
    dialogue_text をそのままプロンプトに投げ、1〜10 の整数スコアひとつを返す。
    """
    dialogue_text = (dialogue_text or "").strip()
    if not dialogue_text:
        return {"score": None, "raw": ""}

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
\"\"\"{dialogue_text}\"\"\"""".strip()

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
            max_new_tokens=8,  # 数字のみ想定
            temperature=0.0,  # 決定的
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

    gen_only = output[inputs.shape[-1] :]
    answer = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

    m = re.search(r"\b(10|[1-9])\b", answer)
    score = int(m.group(1)) if m else None
    return {"score": score, "raw": answer}


# ===== 入出力 =====
input_path = "data/moshi_stage3_old_jchat_clean_tabidachi/transcripts_dialog.jsonl"  # {"audio_path":..., "dialogue_text": "...", ...}
output_path = "evaluated.jsonl"

n_total = 0
n_scored = 0

with open(input_path, "r", encoding="utf-8") as fin, open(
    output_path, "w", encoding="utf-8"
) as fout:
    for line in fin:
        n_total += 1
        obj = json.loads(line)

        dialogue = obj.get("dialogue_text", "")
        if not dialogue:
            obj.update({"score": None, "judge_raw": ""})
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            continue

        result = evaluate_text(dialogue)
        obj.update({"score": result["score"], "judge_raw": result["raw"]})
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        if result["score"] is not None:
            n_scored += 1

print(f"評価結果を {output_path} に保存しました。（{n_scored}/{n_total} 件スコア付与）")
