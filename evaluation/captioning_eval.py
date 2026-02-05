# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluates captioning model outputs using Meteor, Rouge, and Cider metrics.

This script processes a directory of JSON files containing model predictions and ground truth answers,
computes evaluation metrics for specified categories, and saves the results to a CSV file.

Example usage:
    python evaluate_captioning.py \
        --json_dir /path/to/jsons \
        --output_csv /path/to/results.csv \
        --categories avsn avdn

Input Format:
    Each JSON file should contain either:
    - A list of items, or
    - A dict with an "items" key containing a list of items.

    Each item should be a dict with at least:
        - "category": str
        - "answer" or "ground_truth": str
        - "output" or "prediction": str

Output Format:
    A CSV file with columns:
        - file_name
        - Meteor
        - Rouge
        - Cider

Raises:
    Prints warnings for files/items with missing or invalid data.
"""

import argparse
import csv
import json
import os
from glob import glob
from statistics import mean

# Metrics
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from tqdm import tqdm


class FixedMeteor(Meteor):
    """Meteor scorer with robust float parsing."""

    def _read_score(self):
        while True:
            line = self.meteor_p.stdout.readline().strip()
            try:
                return float(line)
            except ValueError:
                continue


def evaluate_captioning(json_dir: str, output_csv: str, categories: list[str]) -> None:
    """
    Evaluates captioning results in JSON files from a directory.

    Args:
        json_dir (str): Directory containing JSON files.
        output_csv (str): Path to save the CSV results.
        categories (list[str]): List of categories to evaluate (e.g., ["avsn", "avdn"]).

    Returns:
        None. Writes results to output_csv.

    Example:
        >>> evaluate_captioning(
        ...     json_dir="/data/jsons",
        ...     output_csv="/data/results.csv",
        ...     categories=["avsn", "avdn"]
        ... )
    """
    json_files = glob(os.path.join(json_dir, "*.json"))
    results = []

    for file in tqdm(json_files, desc="Processing files"):
        print(f"\nEvaluating {os.path.basename(file)} ...")
        with open(file, "r") as f:
            data = json.load(f)

        # Extract list of items
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("items", [data])
        else:
            print(f"⚠️ Invalid format in {file}")
            continue

        sample_metrics = {"Meteor": [], "Rouge": [], "Cider": []}
        meteor_scorer = FixedMeteor()
        cider_scorer = Cider()
        rouge_scorer = Rouge()

        for idx, item in enumerate(tqdm(items, desc="Per-sample", leave=False)):
            if item.get("category") not in categories:
                continue
            answer = (
                str(item.get("answer", item.get("ground_truth", "")))
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            )
            output = (
                str(item.get("output", item.get("prediction", "")))
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            )
            if not answer or not output:
                continue

            gts = {0: [answer]}
            res = {0: [output]}

            try:
                cider_score, _ = cider_scorer.compute_score(gts, res)
                meteor_score, _ = meteor_scorer.compute_score(gts, res)
                rouge_score, _ = rouge_scorer.compute_score(gts, res)
            except Exception as e:
                print(f"Error at sample {idx}: {e}")
                continue

            sample_metrics["Cider"].append(cider_score)
            sample_metrics["Meteor"].append(meteor_score)
            sample_metrics["Rouge"].append(rouge_score)

        # Aggregate per-file means
        if any(len(v) for v in sample_metrics.values()):
            results.append(
                {
                    "file_name": os.path.basename(file),
                    "Meteor": (
                        f"{mean(sample_metrics['Meteor']) * 100:.2f}"
                        if sample_metrics["Meteor"]
                        else "0.00"
                    ),
                    "Rouge": (
                        f"{mean(sample_metrics['Rouge']) * 100:.2f}"
                        if sample_metrics["Rouge"]
                        else "0.00"
                    ),
                    "Cider": (
                        f"{mean(sample_metrics['Cider']):.4f}"
                        if sample_metrics["Cider"]
                        else "0.00"
                    ),
                }
            )
        else:
            print(f"No valid samples in {file}")

    # === SAVE CSV ===
    if results:
        with open(output_csv, "w", newline="") as csvfile:
            fieldnames = ["file_name", "Meteor", "Rouge", "Cider"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n Saved evaluation results to {output_csv}")
    else:
        print("No valid data found.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate captioning results using Meteor, Rouge, and Cider metrics."
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        required=True,
        help="Directory containing JSON files with captioning results.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save the CSV results. Defaults to <json_dir>/seg_caption_evaluation_results.csv",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["avsn", "avdn"],
        help="List of categories to evaluate (default: avsn avdn).",
    )
    args = parser.parse_args()
    output_csv = args.output_csv or os.path.join(
        args.json_dir, "seg_caption_evaluation_results.csv"
    )
    evaluate_captioning(args.json_dir, output_csv, args.categories)


if __name__ == "__main__":
    main()
