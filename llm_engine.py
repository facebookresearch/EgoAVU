# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import transformers

from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline"""

    model_id: str
    jsonl_path: str
    prompt_path: str
    output_path: str
    token_mapping: Dict[str, str]
    max_new_tokens: int = 512
    batch_size: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: Optional[str] = None


class JSONLDataset(Dataset):
    """Dataset for JSONL files with prompt template support"""

    def __init__(
        self, jsonl_path: str, prompt_template: str, token_mapping: Dict[str, str]
    ):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        self.prompt_template = prompt_template
        self.token_mapping = token_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = self._fill_template(item)
        return {"original_data": item, "prompt": prompt, "index": idx}

    def _fill_template(self, data: dict) -> str:
        """Fill prompt template with data from JSONL entry"""
        prompt = self.prompt_template

        tokens = re.findall(r"<(\w+)>", prompt)

        for token in tokens:
            if token in self.token_mapping:
                json_key = self.token_mapping[token]
                value = data.get(json_key, f"[{json_key} not found]")
                prompt = prompt.replace(f"<{token}>", str(value))

        return prompt


def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file"""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_distributed_info():
    """Get distributed information from environment variables set by TorchX fb.dist.ddp"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    return rank, world_size, local_rank, device


def init_distributed():
    """Initialize distributed training if running in multi-process mode"""
    rank, world_size, local_rank, device = get_distributed_info()

    if world_size > 1:
        if not dist.is_initialized():
            # Initialize process group - TorchX should have set the required env vars
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://",
                world_size=world_size,
                rank=rank,
            )
            print(
                f"Initialized distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}"
            )
        else:
            print(
                f"Distributed already initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}"
            )
    else:
        print(
            f"Single process mode: rank={rank}, world_size={world_size}, local_rank={local_rank}"
        )

    return rank, world_size, local_rank, device


def distribute_data(data_points, rank, world_size):
    """Distribute data points across ranks for data parallel processing"""
    total_items = len(data_points)

    if world_size == 1:
        print(f"Single process: processing all {total_items} items")
        return data_points

    items_per_rank = total_items // world_size
    remainder = total_items % world_size

    start_idx = rank * items_per_rank + min(rank, remainder)
    if rank < remainder:
        end_idx = start_idx + items_per_rank + 1
    else:
        end_idx = start_idx + items_per_rank

    rank_data = data_points[start_idx:end_idx]

    print(
        f"Rank {rank}/{world_size}: Processing {len(rank_data)} items (indices {start_idx}-{end_idx-1} of {total_items})"
    )

    return rank_data


def run_inference_worker(config: InferenceConfig):
    """Worker function for each GPU using TorchX distributed setup"""

    rank, world_size, local_rank, device = init_distributed()

    print(
        f"Process info - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}, Device: {device}"
    )

    try:
        print(f"[GPU {rank}] Loading model...")
        pipeline = transformers.pipeline(
            "text-generation",
            model=config.model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": {"": local_rank},  # Force model to specific GPU
            },
        )

        print(f"[GPU {rank}] Model loaded successfully")

        if world_size > 1:
            dist.barrier()
            print(f"[GPU {rank}] All models loaded, proceeding with inference")

        prompt_template = load_prompt_template(config.prompt_path)
        dataset = JSONLDataset(config.jsonl_path, prompt_template, config.token_mapping)

        rank_data = distribute_data(list(range(len(dataset))), rank, world_size)

        print(f"[GPU {rank}] Processing {len(rank_data)} samples")

        results = []
        for idx in tqdm(rank_data, desc=f"GPU {rank}", position=rank):
            item = dataset[idx]

            # Prepare messages
            messages = []
            if config.system_prompt:
                messages.append({"role": "system", "content": config.system_prompt})
            messages.append({"role": "user", "content": item["prompt"]})

            try:
                outputs = pipeline(
                    messages,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                )

                response = outputs[0]["generated_text"][-1]["content"]

                # Combine original data with response
                result = item["original_data"].copy()
                result["response"] = response
                result["prompt_used"] = item["prompt"]
                result["processed_by_rank"] = rank  # Add rank info for debugging
                result["total_ranks"] = world_size

                results.append(result)

            except Exception as e:
                print(f"[GPU {rank}] Error processing sample {idx}: {e}")
                result = item["original_data"].copy()
                result["response"] = f"ERROR: {str(e)}"
                result["prompt_used"] = item["prompt"]
                result["processed_by_rank"] = rank
                result["total_ranks"] = world_size
                results.append(result)

        # Save results for this GPU with rank in filename
        output_path = Path(config.output_path)

        if world_size > 1:
            shard_output = (
                output_path.parent
                / f"{output_path.stem}_rank_{rank}{output_path.suffix}"
            )
        else:
            shard_output = output_path

        with open(shard_output, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"[GPU {rank}] Saved results to {shard_output}")

        # Synchronize all processes before finishing
        if world_size > 1:
            print(f"[GPU {rank}] Waiting for all ranks to complete...")
            dist.barrier()
            print(f"[GPU {rank}] All ranks completed successfully")

    except Exception as e:
        print(f"[GPU {rank}] Fatal error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup - only if we initialized distributed
        if world_size > 1 and dist.is_initialized():
            # Don't call destroy here as TorchX handles cleanup
            pass
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def merge_shard_outputs(output_path: str, world_size: int):
    """Merge outputs from all GPU shards"""
    output_path = Path(output_path)
    all_results = []

    # Read all shard files
    for rank in range(world_size):
        shard_file = (
            output_path.parent / f"{output_path.stem}_rank_{rank}{output_path.suffix}"
        )
        if shard_file.exists():
            with open(shard_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
            # Remove shard file after merging
            shard_file.unlink()

    # Write merged results
    with open(output_path, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Merged {len(all_results)} results into {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU LLM Inference Pipeline")
    parser.add_argument(
        "--model_id", type=str, required=True, help="Model ID from HuggingFace"
    )
    parser.add_argument(
        "--jsonl_path", type=str, required=True, help="Path to input JSONL file"
    )
    parser.add_argument(
        "--prompt_path", type=str, required=True, help="Path to prompt template file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        required=True,
        help="Token mapping in format: token1:key1,token2:key2 (e.g., vidcap:video_caption,audcap:audio_caption)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpfull AI Agent",
        help="System prompt for the model",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs (ignored when using TorchX - for compatibility only)",
    )

    args = parser.parse_args()

    # Parse token mapping
    token_mapping = {}
    for pair in args.mapping.split(","):
        token, key = pair.split(":")
        token_mapping[token.strip()] = key.strip()

    # Get distributed info from environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Detected world_size: {world_size}, rank: {rank}")
    print(f"Token mapping: {token_mapping}")

    print(f"Model path: {args.model_id}")

    # Create config
    config = InferenceConfig(
        model_id=args.model_id,
        jsonl_path=args.jsonl_path,
        prompt_path=args.prompt_path,
        output_path=args.output_path,
        token_mapping=token_mapping,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
    )

    # Run inference - TorchX handles the multiprocessing
    run_inference_worker(config)

    # Only rank 0 merges outputs
    if rank == 0 and world_size > 1:
        print("Rank 0: Merging outputs from all ranks...")
        merge_shard_outputs(args.output_path, world_size)
        print("Rank 0: Merging complete")


if __name__ == "__main__":
    main()
