import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-7.2b")
model = AutoModelForCausalLM.from_pretrained(
    "llm-jp/llm-jp-3-7.2b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


def evaluate_text(text: str) -> dict:
    # 4観点を考慮しつつ、最終的に1〜10の整数1つだけを出力させる
    prompt = f"""
次の対話テキストを、以下の4観点を総合的に考慮して評価してください。
- 一貫性 (Coherence)
- 自然さ (Fluency & Naturalness)
- 関連性 (Relevance)
- 情報量 (Informativeness)

出力は、1〜10の**整数**で表す**総合スコア**を**ひとつだけ**にしてください。
数字以外の文字は一切出力しないでください（例: 7）。

対話テキスト:
\"\"\"{text}\"\"\""""

    tokenized_input = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            tokenized_input,
            max_new_tokens=8,  # 数字だけなので短く
            do_sample=False,  # 決定的に
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

    response = tokenizer.decode(output, skip_special_tokens=True)
    answer = response.replace(prompt, "").strip()

    # 1〜10 の整数のみを抽出
    m = re.search(r"\b(10|[1-9])\b", answer)
    if m:
        return {"score": int(m.group(1))}
    return {"score": None}


# 入出力
input_path = "transcripts.jsonl"
output_path = "evaluated.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(
    output_path, "w", encoding="utf-8"
) as fout:
    for line in fin:
        obj = json.loads(line)
        scores = evaluate_text(obj["text"])
        obj.update(scores)
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"評価結果を {output_path} に保存しました。")
