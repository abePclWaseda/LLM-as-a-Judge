import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# モデル読み込み
tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-7.2b")
model = AutoModelForCausalLM.from_pretrained(
    "llm-jp/llm-jp-3-7.2b",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 評価対象ディレクトリ
input_dir = "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/text_by_espnet_lower/podcast_test/00000-of-00001"

# 出力ファイル
output_path = "dialogue_quality_results.jsonl"


def build_conversation_text(cuts):
    """JSONから会話テキストを復元"""
    conv = []
    for turn in cuts:
        conv.append(f"{turn['speaker']}: {turn['word']}")
    return "\n".join(conv)


def evaluate_dialogue(text):
    """会話を評価"""
    prompt = f"""以下の会話の品質を評価してください。
評価基準:
- 一貫性 (Coherence): 1-5
- 自然さ (Fluency & Naturalness): 1-5
- 関連性 (Relevance): 1-5
- 情報量 (Informativeness): 1-5

会話:
{text}

出力フォーマットはJSONで:
{{
  "coherence": <int>,
  "naturalness": <int>,
  "relevance": <int>,
  "informativeness": <int>,
  "overall": <float>, 
  "comment": "<短いコメント>"
}}"""

    tokenized_input = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            tokenized_input,
            max_new_tokens=300,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.05,
        )[0]
    decoded = tokenizer.decode(output, skip_special_tokens=True)
    # プロンプトを削って応答部分のみ取り出す
    response = decoded[len(prompt) :].strip()
    return response


def main():
    results = []
    files = [
        f for f in os.listdir(input_dir) if f.endswith(".json") or f.startswith("cuts")
    ]
    files.sort()

    for fname in files:
        path = os.path.join(input_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        text = build_conversation_text(data)
        evaluation = evaluate_dialogue(text)

        results.append({"file": fname, "evaluation": evaluation})
        print(f"[DONE] {fname}")

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 評価結果を {output_path} に保存しました")


if __name__ == "__main__":
    main()
