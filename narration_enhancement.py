# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
from pathlib import Path
from typing import List, NamedTuple

import librosa
import numpy as np
import numpy.typing as npt
import pandas as pd
import torchvision
from vllm import LLM, SamplingParams


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)

def split_video_chunks(
    video_path: str, chunk_duration: float = 10.0, target_fps: float = 1.0
) -> List[npt.NDArray]:
    """Split video into chunks at target FPS."""
    video, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
    original_fps = info["video_fps"]

    frame_step = max(1, int(original_fps / target_fps))
    video = video[::frame_step]

    frames_per_chunk = int(chunk_duration * target_fps)
    num_chunks = math.ceil(video.shape[0] / frames_per_chunk)
    num_chunks-=1

    chunks = [
        video[i * frames_per_chunk : (i + 1) * frames_per_chunk].numpy()
        for i in range(num_chunks)
    ]

    print(f"Video: {video.shape[0]} frames -> {num_chunks} chunks of {frames_per_chunk} frames each")
    return chunks


def split_audio_chunks(
    video_path: str, chunk_duration: float = 10.0, sampling_rate: int = 16000
) -> List[npt.NDArray]:
    """Split audio into chunks at 16kHz."""
    audio, _ = librosa.load(video_path, sr=sampling_rate, mono=True)

    samples_per_chunk = int(chunk_duration * sampling_rate)
    num_chunks = math.ceil(audio.shape[0] / samples_per_chunk)
    num_chunks-=1
    chunks = [
        audio[i * samples_per_chunk : (i + 1) * samples_per_chunk]
        for i in range(num_chunks)
    ]


    print(f"Audio: {audio.shape[0]} samples -> {num_chunks} chunks of {samples_per_chunk} samples each")
    return chunks


def format_video_audio_query(
    question: str, video: npt.NDArray, audio: npt.NDArray
) -> QueryResult:
    """Format query with video and audio for Qwen2.5-Omni."""
    prompt = (
        f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"video": video, "audio": [audio]},
            "mm_processor_kwargs": {"use_audio_in_video": True},
        },
        limit_mm_per_prompt={"video": 1, "audio": 1},
    )


def process_video(llm: LLM, video_path: str, args) -> dict:
    """Process a single video: split into chunks and run inference."""
    print(f"Processing: {video_path}")

    video_chunks = split_video_chunks(
        video_path,
        chunk_duration=args.chunk_duration,
        target_fps=args.target_fps
    )
    audio_chunks = split_audio_chunks(
        video_path,
        chunk_duration=args.chunk_duration,
        sampling_rate=16000
    )

    # Ensure matching number of chunks
    n_chunks = min(len(video_chunks), len(audio_chunks))
    video_chunks = video_chunks[:n_chunks]
    audio_chunks = audio_chunks[:n_chunks]

    print(f"\nProcessing {n_chunks} chunks...")

    prompts = [
        ("Describe the video in detail", "video_description"),
        ("Describe the sounds heard in the video in detail", "sound_description"),
        ("Identify all the objects visible in the video", "object_description"),
    ]

    all_queries = []
    for chunk_idx, (v_chunk, a_chunk) in enumerate(zip(video_chunks, audio_chunks)):
        for question, _ in prompts:
            query = format_video_audio_query(question, v_chunk, a_chunk)
            all_queries.append(query)

    print(f"Total queries: {len(all_queries)}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    all_responses = []
    batch_size = args.batch_size
    num_batches = math.ceil(len(all_queries) / batch_size)

    print(f"Processing in {num_batches} batches of size {batch_size}")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_queries))
        batch_queries = all_queries[start_idx:end_idx]

        batch_inputs = [q.inputs for q in batch_queries]

        print(f"  Batch {batch_idx + 1}/{num_batches}: processing {len(batch_inputs)} queries...")
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        batch_responses = [o.outputs[0].text for o in outputs]
        all_responses.extend(batch_responses)

    result = {
        "id": Path(video_path).stem,
        "path": video_path,
        "information": [],
    }

    num_prompts = len(prompts)
    for chunk_idx in range(n_chunks):
        chunk_info = {
            "chunk_id": chunk_idx,
            "start_time": chunk_idx * args.chunk_duration,
            "end_time": (chunk_idx + 1) * args.chunk_duration,
        }

        for prompt_idx, (_, key) in enumerate(prompts):
            response_idx = chunk_idx * num_prompts + prompt_idx
            chunk_info[key] = all_responses[response_idx]

        result["information"].append(chunk_info)

    print(f"Completed: {n_chunks} chunks processed")
    return result

def main(args):
    print(f"Loading video paths from: {args.input_data}")
    df = pd.read_csv(args.input_data)
    video_paths = df['output_path'].to_list()
    print(f"Found {len(video_paths)} videos to process")

    print(f"\nInitializing model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )

    all_results = []
    for idx, video_path in enumerate(video_paths):
        try:
            print(f"\n[{idx+1}/{len(video_paths)}]")
            result = process_video(llm, video_path, args)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR: Failed to process {video_path}")
            print(f"Error: {str(e)}")
            continue

    output_file = Path(args.output_dir) / "all_caption.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    
    print(f"Processing complete!")
    print(f"Processed {len(all_results)}/{len(video_paths)} videos")
    print(f"Results saved to: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Process videos with Qwen2.5-Omni")

    # Required
    parser.add_argument("--model-path", required=True, help="Path to Qwen2.5-Omni model")
    parser.add_argument("--input-data", required=True, help="CSV file with 'output_path' column")

    # Output
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")

    # Chunking
    parser.add_argument("--chunk-duration", type=float, default=10.0, help="Chunk duration (seconds)")
    parser.add_argument("--target-fps", type=float, default=16.0, help="Target FPS for video")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")

    # Model params
    parser.add_argument("--max-model-len", type=int, default=32768, help="Max model length")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Max number of sequences")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Generation params
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per generation")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
