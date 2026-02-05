# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_PROMPT = """Objective:
Act as an impartial grader to evaluate a PREDICTED_ANSWER against a GROUNDING_ANSWER with respect to a QUESTION.

Input:
- QUESTION: {question}
- GROUNDING_ANSWER: {grounding}
- PREDICTED_ANSWER: {prediction}

Instructions for Grading:
1. Compare the PREDICTED_ANSWER to the GROUNDING_ANSWER with respect to the QUESTION.
2. Assign Rating (1–5, integer only):
   - 5: Fully correct, complete, and faithful to the grounding.
   - 4: Mostly correct; minor omissions or inaccuracies.
   - 3: Partially correct; misses important details.
   - 2: Largely incorrect; substantial errors.
   - 1: Incorrect or irrelevant.
3. Briefly explain the rating.

Output Format:
Return valid JSON only:
{
  "rating": <int between 1 and 5>,
  "reason": "<brief explanation>"
}
"""


def load_items(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "items" in data:
        return data["items"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Invalid JSON format in {json_path}")


def run_judge(model, tokenizer, prompt, max_new_tokens, temperature):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content


def parse_judge_output(output):
    try:
        parsed = json.loads(output)
        return int(parsed["rating"])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation using Qwen")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input JSON files",
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Path to output CSV file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-235B-A22B-Instruct-2507",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens for judge output",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0.0 for deterministic)",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    )
    model.eval()

    rows = []

    for file_name in os.listdir(args.input_dir):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(args.input_dir, file_name)
        items = load_items(file_path)

        category_scores = {}

        for item in items:
            category = item["category"]
            grounding = item.get("answer", item.get("ground_truth"))
            prediction = item.get("output", item.get("prediction"))
            question = item.get("question", item.get("question"))

            prompt = JUDGE_PROMPT.format(
                question=question,
                grounding=grounding,
                prediction=prediction,
            )

            judge_output = run_judge(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
            )

            rating = parse_judge_output(judge_output)
            if rating is None:
                continue

            category_scores.setdefault(category, []).append(rating)

        for category, scores in category_scores.items():
            avg_score = sum(scores) / len(scores)
            rows.append(
                {
                    "file_name": file_name,
                    "category": category,
                    "score": round(avg_score, 3),
                }
            )

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "category", "score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved judge scores to {args.output_csv}")


if __name__ == "__main__":
    main()
