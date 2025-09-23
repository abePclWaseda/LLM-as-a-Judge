import os
import json
from pathlib import Path
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# モデル読み込み
tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-7.2b")
model = AutoModelForCausalLM.from_pretrained(
    "llm-jp/llm-jp-3-7.2b",
    device_map="auto",
    torch_dtype=torch.bfloat16,  # transformersではこの引数名でOK（警告は無視可）
)

# 評価対象ディレクトリ（または単一ファイルでも可）
input_dir = "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/text_by_espnet_lower/podcast_test/00000-of-00001"

# 出力ファイル
output_path = "dialogue_quality_results.jsonl"


def build_conversation_text(cuts):
    """LLM-as-a-Judge用に自然な会話テキストを復元"""
    conv = []
    prev_spk = None
    buffer = []

    for turn in cuts:
        spk = turn.get("speaker", "UNK")
        word = turn.get("word", "")

        # 話者が切り替わったら flush
        if spk != prev_spk and buffer:
            conv.append(f"{prev_spk}: {''.join(buffer)}")
            buffer = []

        buffer.append(word)
        prev_spk = spk

    # 最後の発話も追加
    if buffer:
        conv.append(f"{prev_spk}: {''.join(buffer)}")

    return "\n".join(conv)


def evaluate_dialogue(text: str) -> str:
    """会話を評価してモデルの生テキスト応答を返す"""
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
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

    decoded = tokenizer.decode(output, skip_special_tokens=True)
    # プロンプトを削って応答部分のみ
    response = decoded[len(prompt) :].strip()
    return response


def parse_json_loose(s: str):
    """
    モデル応答からJSON部分をゆるく抽出してパース。
    見つからなければ None を返す。
    """
    # いちばん外側の { ... } を雑に抽出
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        # 足りないキーがあれば平均でoverallを補完
        required = ["coherence", "naturalness", "relevance", "informativeness"]
        if "overall" not in obj and all(k in obj for k in required):
            vals = [float(obj[k]) for k in required]
            obj["overall"] = sum(vals) / len(vals)
        return obj
    except Exception:
        return None


def iter_json_files(target: Path):
    """targetがファイルならそれだけ、ディレクトリなら再帰で*.jsonを列挙"""
    if target.is_file() and target.suffix.lower() == ".json":
        yield target
    elif target.is_dir():
        for p in sorted(target.rglob("*.json")):
            yield p
    else:
        raise FileNotFoundError(f"Not found or not supported: {target}")


def main():
    in_path = Path(input_dir)

    sample_out = "restored_samples.txt"
    sample_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f, open(
        sample_out, "w", encoding="utf-8"
    ) as sample_f:
        for json_path in iter_json_files(in_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[SKIP: JSON LOAD ERROR] {json_path} -> {e}")
                continue

            # data が配列（各発話のリスト）である前提
            if not isinstance(data, list):
                print(f"[SKIP: NOT A LIST] {json_path}")
                continue

            text = build_conversation_text(data)

            if sample_count < 3:
                sample_f.write(f"### {json_path.name}\n")
                sample_f.write(text + "\n\n")
                sample_count += 1

            resp_text = evaluate_dialogue(text)

            parsed = parse_json_loose(resp_text)
            record = {
                "file": str(
                    json_path.relative_to(in_path)
                    if in_path.is_dir()
                    else json_path.name
                ),
                "raw": resp_text if parsed is None else None,
                "evaluation": parsed if parsed is not None else None,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[DONE] {json_path}")

    print(f"✅ 評価結果を {output_path} に保存しました")
    print(f"✅ サンプル復元文を {sample_out} に保存しました")


if __name__ == "__main__":
    main()
