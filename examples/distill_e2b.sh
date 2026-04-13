#!/bin/bash
# Example: Distill Gemma 4 E2B into Block AttnRes
#
# Prerequisites:
#   1. Download model weights from HuggingFace
#   2. Place model.safetensors in this directory
#   3. Run: nix develop -c bash examples/distill_e2b.sh

set -euo pipefail

MODEL_PATH="${1:-model.safetensors}"
CONFIG="${2:-e2b}"
STEPS="${3:-1000}"
SEQ_LEN="${4:-512}"
LR="${5:-0.0001}"

echo "=== FerrisRes Gemma 4 Distillation ==="
echo ""
echo "Model:    $MODEL_PATH"
echo "Config:   $CONFIG"
echo "Steps:    $STEPS"
echo "Seq len:  $SEQ_LEN"
echo "LR:       $LR"
echo ""

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    echo ""
    echo "Download from HuggingFace:"
    echo "  wget https://huggingface.co/google/gemma-4-e2b-it/resolve/main/model.safetensors"
    exit 1
fi

# Check system info
echo "--- System Info ---"
nix develop -c cargo run -- info 2>/dev/null || echo "(Could not get system info)"
echo ""

# Run distillation
echo "--- Starting Distillation ---"
nix develop -c cargo run -- distill \
    --model-path "$MODEL_PATH" \
    --config "$CONFIG" \
    --steps "$STEPS" \
    --seq-len "$SEQ_LEN" \
    --lr "$LR" \
    --temperature 2.0 \
    --output "distilled_${CONFIG}" \
    --log-every 10

echo ""
echo "=== Results ==="

if [ -f "distilled_${CONFIG}.bin.loss_curve.csv" ]; then
    echo "Loss curve: distilled_${CONFIG}.bin.loss_curve.csv"
    echo ""
    echo "First 5 entries:"
    head -6 "distilled_${CONFIG}.bin.loss_curve.csv"
    echo ""
    echo "Last 5 entries:"
    tail -5 "distilled_${CONFIG}.bin.loss_curve.csv"
fi

echo ""
echo "Block Summary parameters:"
ls -la distilled_${CONFIG}.bin.block_summary_*.bin 2>/dev/null || echo "(none)"

echo ""
echo "Done!"
