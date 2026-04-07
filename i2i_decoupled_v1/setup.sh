#!/bin/bash
# setup.sh — Install environment for laptop OR DGX
# Usage:
#   bash setup.sh          # auto-detect GPU
#   bash setup.sh --cpu    # force CPU-only (laptop without CUDA)
#   bash setup.sh --dgx    # force DGX cu113 build

set -e

MODE="auto"
if [[ "$1" == "--cpu" ]]; then MODE="cpu"; fi
if [[ "$1" == "--dgx" ]]; then MODE="dgx"; fi

echo "=== Illumination Diffusion Environment Setup ==="
echo "Mode: $MODE"

# Detect CUDA
if [[ "$MODE" == "auto" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1)
        echo "Detected CUDA major version: $CUDA_VER"
        if [[ "$CUDA_VER" -ge "11" ]]; then
            MODE="dgx"
        else
            MODE="cpu"
        fi
    else
        echo "No GPU detected, using CPU mode"
        MODE="cpu"
    fi
fi

# Install PyTorch first (must precede diffusers)
if [[ "$MODE" == "dgx" ]]; then
    echo "[DGX] Installing torch 1.11 + cu113..."
    pip install torch==1.11.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113 --quiet
    pip install torchvision==0.12.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113 --quiet
else
    echo "[CPU/Laptop] Installing torch 1.11 (CPU)..."
    pip install torch==1.11.0 torchvision==0.12.0 --quiet
fi

# Install remaining dependencies
echo "Installing remaining dependencies..."
pip install \
    diffusers==0.14.0 \
    huggingface_hub==0.25.2 \
    accelerate==0.18.0 \
    transformers==4.27.4 \
    wandb==0.14.2 \
    Pillow==9.5.0 \
    numpy==1.23.5 \
    tqdm==4.65.0 \
    matplotlib==3.7.1 \
    packaging==23.1 \
    PyYAML>=6.0 \
    --quiet

echo ""
echo "=== Installation complete ==="
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
echo ""
echo "Next steps:"
echo "  1. python scripts/prepare_dataset.py --data_root ./data --create_dummy"
echo "  2. python train.py --config configs/laptop_debug.yaml   # laptop verify"
echo "  3. python train.py --config configs/dgx_train.yaml      # DGX training"
