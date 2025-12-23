#!/bin/bash
set -e
cd /workspace/videomask

source .venv/bin/activate

export HF_HOME=/root/.cache/huggingface
export HF_TOKEN=${HF_TOKEN:-""}
export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}

echo "Using python: $(which python)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python -m streamlit run conceptops/demo/app_streamlit.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false \
  --server.enableCORS false \
  --server.enableXsrfProtection false
