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
    prompt = f"""
次の対話テキストについて、以下の4つの観点で1〜10の整数スコアを出してください。
必ずJSON形式で、整数値のみを出力してください。

観点:
- 一貫性 (Coherence)
- 自然さ (Fluency & Naturalness)
- 関連性 (Relevance)
- 情報量 (Informativeness)

出力例:
{{"Coherence": 8, "Fluency": 7, "Relevance": 9, "Informativeness": 6}}

対話テキスト:
\"\"\"{text}\"\"\""""

    # あなたの提示スタイル（encode → generate）
    tokenized_input = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            tokenized_input,
            max_new_tokens=200,
            do_sample=False,  # 評価なので確定的に
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

    response = tokenizer.decode(output, skip_special_tokens=True)

    # プロンプトを取り除く
    answer = response.replace(prompt, "").strip()

    # JSON部分を抽出
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "Coherence": None,
        "Fluency": None,
        "Relevance": None,
        "Informativeness": None,
    }


# 書き起こし結果の読み込み
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
