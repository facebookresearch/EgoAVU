import json
import os
import re
import csv

def parse_paths_to_csv(json_path, output_csv):
    with open(json_path, "r") as f:
        paths = json.load(f)
    paths = [item['audios'][0] for item in paths]
    rows = []

    for path in paths:
        filename = os.path.basename(path)
        file_id = os.path.splitext(filename)[0]

        # Extract part index
        match = re.search(r"part(\d+)", filename)
        if not match:
            continue

        part_id = int(match.group(1))
        start_time = part_id * 360

        # Check if this is a 1-minute chunk
        if "1min" in path:
            end_time = start_time + 60
        else:
            end_time = start_time + 360

        rows.append({
            "id": file_id.split('_')[0],
            "start_time": start_time,
            "end_time": end_time,
            "split": "train"
        })

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "start_time", "end_time", "split"])
        writer.writeheader()
        writer.writerows(rows)

# Example usage
parse_paths_to_csv("/mnt/xr_core_ai_asl_llm/tree/users/ashish/Multimodal_source/training_data.json", "output_segments.csv")
