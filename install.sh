#!/usr/bin/env bash
set -e
# пример установки для CUDA 12.8
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install trackeval==1.3.0 --no-deps

pip check
