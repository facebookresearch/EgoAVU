# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Distributed video and audio processing with Qwen2.5-Omni model using vLLM.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import numpy.typing as npt
import torch
import torch.distributed as dist
import torchaudio
import torchvision
import pandas as pd

from vllm import LLM, SamplingParams

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Rank %(rank)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RankFilter(logging.Filter):
    def filter(self, record):
        record.rank = int(os.environ.get("RANK", 0))
        return True


logger = logging.getLogger(__name__)
logger.addFilter(RankFilter())

# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)

# ----------------------------------------------------------------------------
# Distributed setup
# ----------------------------------------------------------------------------


def get_distributed_info():
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
    rank, world_size, local_rank, device = get_distributed_info()

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        logger.info(
            f"Initialized distributed training: world_size={world_size}, "
            f"local_rank={local_rank}, device={device}"
        )
    else:
        logger.info(f"Single process mode: device={device}")

    return rank, world_size, local_rank, device


def distribute_data(data: List, rank: int, world_size: int) -> List:
    if world_size == 1:
        return data

    n = len(data)
    per_rank = n // world_size
    rem = n % world_size

    start = rank * per_rank + min(rank, rem)
    end = start + per_rank + (1 if rank < rem else 0)

    return data[start:end]

# ----------------------------------------------------------------------------
# Video / Audio processing
# ----------------------------------------------------------------------------


def video_to_ndarrays_chunks(
    path: str, chunk_duration: float, max_chunks: Optional[int]
) -> List[npt.NDArray]:
    video, _, info = torchvision.io.read_video(path, pts_unit="sec")
    fps = info["video_fps"]

    frames_per_chunk = int(chunk_duration * fps)
    total_frames = video.shape[0]
    num_chunks = math.ceil(total_frames / frames_per_chunk)

    if max_chunks is not None:
        num_chunks = min(num_chunks, max_chunks)

    return [
        video[i * frames_per_chunk : (i + 1) * frames_per_chunk].numpy()
        for i in range(num_chunks)
    ]


def get_audio_chunks(
    path: str,
    chunk_duration: float,
    sampling_rate: int = 16000,
    max_chunks: Optional[int] = None,
) -> List[npt.NDArray]:
    audio, sr = torchaudio.load(path)

    if sr != sampling_rate:
        audio = torchaudio.transforms.Resample(sr, sampling_rate)(audio)

    audio = audio.mean(0)
    samples_per_chunk = int(chunk_duration * sampling_rate)
    num_chunks = math.ceil(audio.shape[0] / samples_per_chunk)

    if max_chunks is not None:
        num_chunks = min(num_chunks, max_chunks)

    return [
        audio[i * samples_per_chunk : (i + 1) * samples_per_chunk].numpy()
        for i in range(num_chunks)
    ]


def has_sound(waveform: torch.Tensor, threshold: float) -> bool:
    return torch.sqrt(torch.mean(waveform**2)).item() > threshold

# ----------------------------------------------------------------------------
# Query formatting
# ----------------------------------------------------------------------------


def format_video_audio_query(
    question: str, video: npt.NDArray, audio: npt.NDArray
) -> QueryResult:
    prompt = (
        f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"video": video, "audio": audio},
            "mm_processor_kwargs": {"use_audio_in_video": True},
        },
        limit_mm_per_prompt={"video": 1, "audio": 1},
    )


def batchify(xs: List, bs: int) -> List[List]:
    return [xs[i : i + bs] for i in range(0, len(xs), bs)]

# ----------------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------------


def process_video_audio(
    llm: LLM, video_path: str, args, rank: int, world_size: int
) -> Dict[str, Any]:

    video_chunks = video_to_ndarrays_chunks(
        video_path, args.chunk_duration, args.max_chunks
    )
    audio_chunks = get_audio_chunks(
        video_path, args.chunk_duration, max_chunks=args.max_chunks
    )

    n_chunks = min(len(video_chunks), len(audio_chunks))
    video_chunks = video_chunks[:n_chunks]
    audio_chunks = audio_chunks[:n_chunks]

    prompts = [
        ("Describe the video in detail", "video_description"),
        ("Describe the sounds heard in the video in detail", "sound_description"),
        ("Identify all the objects visible in the video", "object_description"),
    ]

    queries = []
    for v, a in zip(video_chunks, audio_chunks):
        for q, _ in prompts:
            queries.append(format_video_audio_query(q, v, a))

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )

    outputs = []
    for batch in batchify(queries, args.batch_size):
        batch_inputs = [q.inputs for q in batch]
        res = llm.generate(batch_inputs, sampling_params=sampling_params)
        outputs.extend([o.outputs[0].text for o in res])

    result = {
        "id": Path(video_path).stem,
        "path": video_path,
        "information": [],
    }

    num_q = len(prompts)

    for i in range(n_chunks):
        chunk_info = {
            "start_time": i * args.chunk_duration,
            "end_time": (i + 1) * args.chunk_duration,
        }

        for j, (_, key) in enumerate(prompts):
            chunk_info[key] = outputs[i * num_q + j]

        result["information"].append(chunk_info)

    return result

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main(args):
    rank, world_size, _, _ = init_distributed()

    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        distributed_executor_backend="external_launcher"
        if world_size > 1
        else None,
    )
    data = pd.read_csv(args.input_data)['output_path'].to_list()

    data = distribute_data(data, rank, world_size)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for video_path in data:
        out = process_video_audio(llm, video_path, args, rank, world_size)
        out_file = Path(args.output_dir) / f"{Path(video_path).stem}_rank{rank}.json"
        with open(out_file, "w") as f:
            for entry in out:
                f.write(json.dumps(entry)+'\n')

    if world_size > 1:
        dist.barrier()


# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--input-data", required=True)
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--chunk-duration", type=float, default=10.0)
    p.add_argument("--max-chunks", type=int)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
