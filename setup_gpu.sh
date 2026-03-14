#!/bin/bash
# ============================================================
# GPU Machine Setup Script for REG_ML
# Tested on: Ubuntu with NVIDIA RTX 3090
# Usage: bash setup_gpu.sh
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== REG_ML GPU Setup ===${NC}\n"

# ---- 1. Check NVIDIA Driver & GPU ----
echo -e "${YELLOW}[1/6] Checking NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo -e "${GREEN}  -> NVIDIA driver OK${NC}"
else
    echo -e "${RED}  -> nvidia-smi not found! Install NVIDIA drivers first:${NC}"
    echo "     sudo apt install nvidia-driver-535"
    exit 1
fi

# ---- 2. Check CUDA ----
echo -e "\n${YELLOW}[2/6] Checking CUDA...${NC}"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo -e "${GREEN}  -> CUDA OK${NC}"
else
    echo -e "${YELLOW}  -> nvcc not found (OK if PyTorch bundles its own CUDA runtime)${NC}"
fi

# ---- 3. Check Python ----
echo -e "\n${YELLOW}[3/6] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}  -> Python not found! Install with: sudo apt install python3 python3-pip python3-venv${NC}"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1)
echo "  $PY_VERSION"

# Check minimum version (3.10+)
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 10 ]; then
    echo -e "${RED}  -> Python 3.10+ required, got $PY_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}  -> Python OK${NC}"

# ---- 4. Create virtual environment ----
echo -e "\n${YELLOW}[4/6] Setting up virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    echo -e "${GREEN}  -> Created .venv${NC}"
else
    echo -e "${GREEN}  -> .venv already exists${NC}"
fi

source .venv/bin/activate
echo "  Using: $(which python)"

# ---- 5. Install dependencies ----
echo -e "\n${YELLOW}[5/6] Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel -q

# Install PyTorch with CUDA 12.1 support first
echo "  Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# Install remaining dependencies
echo "  Installing project dependencies..."
pip install -r requirements.txt -q

echo -e "${GREEN}  -> Dependencies installed${NC}"

# ---- 6. Verify GPU access from Python ----
echo -e "\n${YELLOW}[6/6] Verifying PyTorch CUDA access...${NC}"
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:    {torch.version.cuda}')
    print(f'  GPU:             {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:            {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('  WARNING: CUDA not available! Training will fall back to CPU.')
"

# ---- Done ----
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate environment:  source .venv/bin/activate"
echo "  2. Set your API key:      cp .env.example .env && nano .env"
echo "  3. Start training:        python run.py train --config config/gpu.yaml"
echo ""
