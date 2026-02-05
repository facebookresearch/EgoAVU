#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# =============================================================================
# EgoAVU Pipeline Script
# =============================================================================
# This script runs the complete EgoAVU pipeline:
# 1. Split video+audio into segments using utils/split_video.py
# 2. Generate captions using narration_enhancement.py
# 3. Generate MCG (Multimodal Context Graph) using llm_engine.py + prompt_mcg.txt
# 4. Generate combined AV narration using llm_engine.py + prompt_combine_av_caption.txt
# 5. Generate QA pairs using llm_engine.py with various QA prompts
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration - Modify these variables as needed
# -----------------------------------------------------------------------------

# Paths
CSV_FILE="${CSV_FILE:-./utils/sample_vid.csv}"
INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-./media/input}"
SPLIT_OUTPUT_DIR="${SPLIT_OUTPUT_DIR:-./media/split}"
CAPTION_OUTPUT_DIR="${CAPTION_OUTPUT_DIR:-./outputs/captions}"
MCG_OUTPUT_DIR="${MCG_OUTPUT_DIR:-./outputs/mcg}"
AV_NARRATION_OUTPUT_DIR="${AV_NARRATION_OUTPUT_DIR:-./outputs/av_narration}"
QA_OUTPUT_DIR="${QA_OUTPUT_DIR:-./outputs/qa}"

# Model paths
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}"
LLM_MODEL_ID="${LLM_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"

# Prompt paths
PROMPT_MCG="${PROMPT_MCG:-./prompts/prompt_mcg.txt}"
PROMPT_COMBINE_AV="${PROMPT_COMBINE_AV:-./prompts/prompt_combine_av_caption.txt}"
PROMPT_QA_DIR="${PROMPT_QA_DIR:-./prompts/qa_generation}"

# Processing parameters
SPLIT_WORKERS="${SPLIT_WORKERS:-4}"
CHUNK_DURATION="${CHUNK_DURATION:-10.0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
NUM_GPUS="${NUM_GPUS:-4}"

# LLM inference parameters
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

check_file_exists() {
    if [[ ! -f "$1" ]]; then
        log_error "File not found: $1"
        exit 1
    fi
}

check_dir_exists() {
    if [[ ! -d "$1" ]]; then
        log_error "Directory not found: $1"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Step 1: Split Videos
# -----------------------------------------------------------------------------

split_videos() {
    log_info "=== Step 1: Splitting videos into ${CHUNK_DURATION}s segments ==="

    check_file_exists "$CSV_FILE"
    check_dir_exists "$INPUT_VIDEO_DIR"

    python utils/split_video.py \
        "$CSV_FILE" \
        "$INPUT_VIDEO_DIR" \
        "$SPLIT_OUTPUT_DIR" \
        --workers "$SPLIT_WORKERS"

    log_info "Video splitting completed. Output saved to: $SPLIT_OUTPUT_DIR"
}

# -----------------------------------------------------------------------------
# Step 2: Generate Captions with Narration Enhancement
# -----------------------------------------------------------------------------

generate_captions() {
    log_info "=== Step 2: Generating captions using narration_enhancement.py ==="

    # Find the output CSV from split step
    SPLIT_CSV=$(find "$SPLIT_OUTPUT_DIR" -name "*_output.csv" | head -n 1)

    if [[ -z "$SPLIT_CSV" ]]; then
        log_error "Could not find split output CSV in $SPLIT_OUTPUT_DIR"
        exit 1
    fi

    log_info "Using split CSV: $SPLIT_CSV"

    mkdir -p "$CAPTION_OUTPUT_DIR"

    python narration_enhancement.py \
        --model-path "$QWEN_MODEL_PATH" \
        --input-data "$SPLIT_CSV" \
        --output-dir "$CAPTION_OUTPUT_DIR" \
        --chunk-duration "$CHUNK_DURATION" \
        --batch-size "$BATCH_SIZE" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --trust-remote-code

    # Merge all caption JSONs into a single JSONL file
    MERGED_CAPTIONS="$CAPTION_OUTPUT_DIR/merged_captions.jsonl"
    log_info "Merging caption files into: $MERGED_CAPTIONS"
    > "$MERGED_CAPTIONS"
    for json_file in "$CAPTION_OUTPUT_DIR"/*.json; do
        if [[ -f "$json_file" ]]; then
            cat "$json_file" >> "$MERGED_CAPTIONS"
        fi
    done

    log_info "Caption generation completed. Output saved to: $CAPTION_OUTPUT_DIR"
}

# -----------------------------------------------------------------------------
# Step 3: Generate MCG (Multimodal Context Graph)
# -----------------------------------------------------------------------------

generate_mcg() {
    log_info "=== Step 3: Generating Multimodal Context Graph (MCG) ==="

    check_file_exists "$PROMPT_MCG"

    MERGED_CAPTIONS="$CAPTION_OUTPUT_DIR/merged_captions.jsonl"
    check_file_exists "$MERGED_CAPTIONS"

    mkdir -p "$MCG_OUTPUT_DIR"

    # Run LLM engine with MCG prompt
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        llm_engine.py \
        --model_id "$LLM_MODEL_ID" \
        --jsonl_path "$MERGED_CAPTIONS" \
        --prompt_path "$PROMPT_MCG" \
        --output_path "$MCG_OUTPUT_DIR/mcg_output.jsonl" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --response_key "mcg"

    log_info "MCG generation completed. Output saved to: $MCG_OUTPUT_DIR/mcg_output.jsonl"
}

# -----------------------------------------------------------------------------
# Step 4: Generate Combined Audio-Visual Narration
# -----------------------------------------------------------------------------

generate_av_narration() {
    log_info "=== Step 4: Generating Combined Audio-Visual Narration ==="

    check_file_exists "$PROMPT_COMBINE_AV"

    MCG_OUTPUT="$MCG_OUTPUT_DIR/mcg_output.jsonl"
    check_file_exists "$MCG_OUTPUT"

    mkdir -p "$AV_NARRATION_OUTPUT_DIR"

    # Run LLM engine with combined AV prompt (uses MCG + original input)
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        llm_engine.py \
        --model_id "$LLM_MODEL_ID" \
        --jsonl_path "$MCG_OUTPUT" \
        --prompt_path "$PROMPT_COMBINE_AV" \
        --output_path "$AV_NARRATION_OUTPUT_DIR/av_narration_output.jsonl" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --response_key "av_caption"

    log_info "AV Narration generation completed. Output saved to: $AV_NARRATION_OUTPUT_DIR/av_narration_output.jsonl"
}

# -----------------------------------------------------------------------------
# Step 5: Generate QA Pairs
# -----------------------------------------------------------------------------

generate_qa() {
    log_info "=== Step 5: Generating QA Pairs ==="

    check_dir_exists "$PROMPT_QA_DIR"

    AV_NARRATION_OUTPUT="$AV_NARRATION_OUTPUT_DIR/av_narration_output.jsonl"
    check_file_exists "$AV_NARRATION_OUTPUT"

    mkdir -p "$QA_OUTPUT_DIR"

    # Define QA types and their prompts
    declare -A QA_PROMPTS=(
        ["avdn"]="$PROMPT_QA_DIR/prompt_avdn.txt"
        ["avh_action"]="$PROMPT_QA_DIR/prompt_avh_action.txt"
        ["avh_object"]="$PROMPT_QA_DIR/prompt_avh_object.txt"
        ["avh_sound"]="$PROMPT_QA_DIR/prompt_avh_sound.txt"
        ["ssa"]="$PROMPT_QA_DIR/prompt_ssa.txt"
        ["tr_before_after"]="$PROMPT_QA_DIR/prompt_tr_before_after.txt"
        ["tr_event_ordering"]="$PROMPT_QA_DIR/prompt_tr_event_ordering.txt"
    )

    # Generate QA for each type
    for qa_type in "${!QA_PROMPTS[@]}"; do
        prompt_file="${QA_PROMPTS[$qa_type]}"

        if [[ ! -f "$prompt_file" ]]; then
            log_error "Prompt file not found: $prompt_file, skipping $qa_type"
            continue
        fi

        log_info "Generating QA type: $qa_type"

        torchrun \
            --nproc_per_node="$NUM_GPUS" \
            llm_engine.py \
            --model_id "$LLM_MODEL_ID" \
            --jsonl_path "$AV_NARRATION_OUTPUT" \
            --prompt_path "$prompt_file" \
            --output_path "$QA_OUTPUT_DIR/qa_${qa_type}.jsonl" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --response_key "qa_${qa_type}" \
            --qa_mode \
            --qa_keys "av_caption" "mcg"

        log_info "QA type $qa_type completed."
    done

    log_info "QA generation completed. Outputs saved to: $QA_OUTPUT_DIR/"
}

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run the complete EgoAVU pipeline or individual steps."
    echo ""
    echo "Options:"
    echo "  --all              Run all steps (default)"
    echo "  --split            Run only video splitting (Step 1)"
    echo "  --caption          Run only caption generation (Step 2)"
    echo "  --mcg              Run only MCG generation (Step 3)"
    echo "  --av-narration     Run only AV narration generation (Step 4)"
    echo "  --qa               Run only QA generation (Step 5)"
    echo "  --from-mcg         Run from MCG generation onwards (Steps 3-5)"
    echo "  --from-av          Run from AV narration onwards (Steps 4-5)"
    echo "  --help             Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  CSV_FILE              Path to input CSV file (default: ./utils/sample_vid.csv)"
    echo "  INPUT_VIDEO_DIR       Directory containing input videos (default: ./media/input)"
    echo "  SPLIT_OUTPUT_DIR      Output directory for split videos (default: ./media/split)"
    echo "  CAPTION_OUTPUT_DIR    Output directory for captions (default: ./outputs/captions)"
    echo "  MCG_OUTPUT_DIR        Output directory for MCG (default: ./outputs/mcg)"
    echo "  AV_NARRATION_OUTPUT_DIR Output directory for AV narration (default: ./outputs/av_narration)"
    echo "  QA_OUTPUT_DIR         Output directory for QA pairs (default: ./outputs/qa)"
    echo "  QWEN_MODEL_PATH       Path to Qwen model (default: Qwen/Qwen2.5-Omni-7B)"
    echo "  LLM_MODEL_ID          LLM model ID (default: meta-llama/Llama-3.1-8B-Instruct)"
    echo "  NUM_GPUS              Number of GPUs for distributed processing (default: 4)"
    echo "  CHUNK_DURATION        Duration of video chunks in seconds (default: 10.0)"
    echo ""
    echo "Example:"
    echo "  $0 --all"
    echo "  CSV_FILE=./my_videos.csv INPUT_VIDEO_DIR=./videos $0 --all"
    echo "  $0 --from-mcg  # Resume pipeline from MCG generation"
}

main() {
    local run_split=false
    local run_caption=false
    local run_mcg=false
    local run_av=false
    local run_qa=false
    local run_all=true

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                run_all=true
                shift
                ;;
            --split)
                run_split=true
                run_all=false
                shift
                ;;
            --caption)
                run_caption=true
                run_all=false
                shift
                ;;
            --mcg)
                run_mcg=true
                run_all=false
                shift
                ;;
            --av-narration)
                run_av=true
                run_all=false
                shift
                ;;
            --qa)
                run_qa=true
                run_all=false
                shift
                ;;
            --from-mcg)
                run_mcg=true
                run_av=true
                run_qa=true
                run_all=false
                shift
                ;;
            --from-av)
                run_av=true
                run_qa=true
                run_all=false
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    log_info "=========================================="
    log_info "Starting EgoAVU Pipeline"
    log_info "=========================================="
    log_info "Configuration:"
    log_info "  CSV_FILE: $CSV_FILE"
    log_info "  INPUT_VIDEO_DIR: $INPUT_VIDEO_DIR"
    log_info "  SPLIT_OUTPUT_DIR: $SPLIT_OUTPUT_DIR"
    log_info "  CAPTION_OUTPUT_DIR: $CAPTION_OUTPUT_DIR"
    log_info "  MCG_OUTPUT_DIR: $MCG_OUTPUT_DIR"
    log_info "  AV_NARRATION_OUTPUT_DIR: $AV_NARRATION_OUTPUT_DIR"
    log_info "  QA_OUTPUT_DIR: $QA_OUTPUT_DIR"
    log_info "  CHUNK_DURATION: ${CHUNK_DURATION}s"
    log_info "  NUM_GPUS: $NUM_GPUS"
    log_info "=========================================="

    if $run_all; then
        split_videos
        generate_captions
        generate_mcg
        generate_av_narration
        generate_qa
    else
        $run_split && split_videos
        $run_caption && generate_captions
        $run_mcg && generate_mcg
        $run_av && generate_av_narration
        $run_qa && generate_qa
    fi

    log_info "=========================================="
    log_info "Pipeline completed successfully!"
    log_info "=========================================="
}

main "$@"
