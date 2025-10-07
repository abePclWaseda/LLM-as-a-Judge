# tools/llm_as_a_judge.py
import argparse
import json
import os
import re
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "llm-jp/llm-jp-3.1-13b-instruct4"


def _to_dtype(name: str):
    name = (name or "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def evaluate_text(
    model,
    tokenizer,
    dialogue_text: str,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
) -> dict:
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

    gen_only = output[inputs.shape[-1] :]
    answer = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

    m = re.search(r"\b(10|[1-9])\b", answer)
    score = int(m.group(1)) if m else None
    return {"score": score, "raw": answer}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge: JSONLを読み、対話テキストを1〜10で採点"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="入力JSONL（各行は {.., 'dialogue_text': '...'}）",
    )
    parser.add_argument(
        "--output", required=True, help="出力JSONL（score, judge_rawを付与）"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Faceモデル名（default: {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='transformersのdevice_map（default: "auto"）',
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help='bfloat16|float16|float32 等（default: "bfloat16"）',
    )
    parser.add_argument("--seed", type=int, default=0, help="乱数シード（default: 0）")
    parser.add_argument(
        "--max-new-tokens", type=int, default=8, help="生成トークン数（default: 8）"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="温度（default: 0.0; 決定的に）"
    )
    parser.add_argument("--overwrite", action="store_true", help="出力の上書きを許可")
    args = parser.parse_args()

    # 入出力チェック
    if not os.path.exists(args.input):
        print(f"[Error] input not found: {args.input}", file=sys.stderr)
        return 2

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.output) and not args.overwrite:
        print(
            f"[Error] output exists: {args.output} (use --overwrite)", file=sys.stderr
        )
        return 3

    # 乱数シード
    torch.manual_seed(args.seed)

    # モデル/トークナイザ
    print(f"[Info] Loading model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=_to_dtype(args.dtype),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ストリーム処理
    n_total = 0
    n_scored = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            n_total += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[Warn] skip (json error): line#{n_total}: {e}", file=sys.stderr)
                continue

            dialogue = obj.get("dialogue_text", "") or ""
            if not dialogue.strip():
                obj.update({"score": None, "judge_raw": ""})
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            try:
                result = evaluate_text(
                    model=model,
                    tokenizer=tokenizer,
                    dialogue_text=dialogue,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                obj.update({"score": result["score"], "judge_raw": result["raw"]})
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                if result["score"] is not None:
                    n_scored += 1
            except Exception as e:
                print(
                    f"[Warn] generation failed at line#{n_total}: {e}", file=sys.stderr
                )
                obj.update({"score": None, "judge_raw": ""})
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if n_total % 20 == 0:
                print(f"[Info] processed={n_total} scored={n_scored}", flush=True)

    print(f"[Done] saved to {args.output} (scored {n_scored}/{n_total})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
