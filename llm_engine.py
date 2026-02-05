# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class InferenceConfig:
    model_id: str
    jsonl_path: str
    prompt_path: str
    output_path: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: Optional[str] = None
    response_key: str = "response"

    qa_mode: bool = False
    qa_keys: Optional[List[str]] = None


class JSONLDataset(Dataset):
    def __init__(self, jsonl_path: str):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def init_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

    return rank, world_size, local_rank


def distribute_indices(n, rank, world_size):
    if world_size == 1:
        return range(n)

    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return range(start, end)


def stringify_information(info_list, keys):
    lines = []
    for i, item in enumerate(info_list, 1):
        lines.append(f"Segment {i}:")
        for k in keys:
            if k in item:
                lines.append(f"{k}: {item[k]}")
        lines.append("")
    return "\n".join(lines).strip()


def run_inference_worker(config: InferenceConfig):
    rank, world_size, local_rank = init_distributed()

    pipe = transformers.pipeline(
        "text-generation",
        model=config.model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": {"": local_rank},
        },
    )

    prompt_template = load_prompt_template(config.prompt_path)
    dataset = JSONLDataset(config.jsonl_path)
    indices = distribute_indices(len(dataset), rank, world_size)

    results = []

    for idx in tqdm(indices, desc=f"GPU {rank}", position=rank):
        entry = json.loads(json.dumps(dataset[idx]))

        if config.qa_mode:
            assert config.qa_keys, "qa_keys must be provided when qa_mode is enabled"

            info_str = stringify_information(
                entry.get("information", []),
                config.qa_keys,
            )

            prompt = prompt_template.replace("<input>", info_str)

            messages = []
            if config.system_prompt:
                messages.append({"role": "system", "content": config.system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                out = pipe(
                    messages,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                )
                entry[config.response_key] = out[0]["generated_text"][-1]["content"]
            except Exception as e:
                entry[config.response_key] = f"ERROR: {str(e)}"

        else:
            for info in entry.get("information", []):
                prompt = prompt_template.replace(
                    "<input>", json.dumps(info, ensure_ascii=False)
                )

                messages = []
                if config.system_prompt:
                    messages.append({"role": "system", "content": config.system_prompt})
                messages.append({"role": "user", "content": prompt})

                try:
                    out = pipe(
                        messages,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        do_sample=True,
                    )
                    info[config.response_key] = out[0]["generated_text"][-1]["content"]
                except Exception as e:
                    info[config.response_key] = f"ERROR: {str(e)}"

        entry["processed_by_rank"] = rank
        entry["total_ranks"] = world_size
        results.append(entry)

    out_path = Path(config.output_path)
    shard = (
        out_path.parent / f"{out_path.stem}_rank_{rank}{out_path.suffix}"
        if world_size > 1
        else out_path
    )

    with open(shard, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if world_size > 1:
        dist.barrier()


def merge_shards(output_path, world_size):
    p = Path(output_path)
    merged = []

    for r in range(world_size):
        shard = p.parent / f"{p.stem}_rank_{r}{p.suffix}"
        if shard.exists():
            with open(shard, "r", encoding="utf-8") as f:
                merged.extend(json.loads(l) for l in f if l.strip())
            shard.unlink()

    with open(p, "w", encoding="utf-8") as f:
        for m in merged:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", required=True)
    parser.add_argument("--jsonl_path", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output_path", required=True)

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument("--response_key", default="response")

    parser.add_argument("--qa_mode", action="store_true")
    parser.add_argument("--qa_keys", nargs="+", default=None)

    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    config = InferenceConfig(**vars(args))
    run_inference_worker(config)

    if rank == 0 and world_size > 1:
        merge_shards(args.output_path, world_size)


if __name__ == "__main__":
    main()
