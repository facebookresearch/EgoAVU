# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import subprocess
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple


def split_video(
    video_path: Path, start: int, end: int, output_path: Path
) -> Tuple[str, bool, str, Path]:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration = end - start

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return (str(video_path), True, "", output_path)
        else:
            return (str(video_path), False, result.stderr, output_path)

    except Exception as e:
        return (str(video_path), False, str(e), output_path)


def process_csv(csv_file: str, input_dir: str, output_dir: str, max_workers: int):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tasks = []
    rows = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["id"]
            start = int(row["start_time"])
            end = int(row["end_time"])
            split = row["split"]

            video_file = input_path / f"{video_id}.mp4"
            output_name = f"{video_id}_{start}_{end}.mp4"
            output_file = output_path / split / output_name

            tasks.append((video_file, start, end, output_file))
            rows.append(row)

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(split_video, *task): i for i, task in enumerate(tasks)
        }

        for future in as_completed(futures):
            idx = futures[future]
            video_path, success, error, out_path = future.result()
            results[idx] = str(out_path.absolute()) if success else ""

            if success:
                print(f"{Path(video_path).name}")
            else:
                print(f"{Path(video_path).name}: {error}")

    # Write updated CSV to output_dir
    output_csv = output_path / f"{Path(csv_file).stem}_output.csv"
    with open(output_csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) + ["output_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            row["output_path"] = results.get(i, "")
            writer.writerow(row)

    print(f"\nOutput CSV saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split videos based on CSV timestamps")
    parser.add_argument(
        "csv_file", help="Path to CSV file with video split information"
    )
    parser.add_argument("input_dir", help="Directory containing input video files")
    parser.add_argument("output_dir", help="Directory to save split video files")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    args = parser.parse_args()

    process_csv(args.csv_file, args.input_dir, args.output_dir, args.workers)
