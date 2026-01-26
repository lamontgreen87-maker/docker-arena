#!/bin/bash

# Stop on error
set -e

echo "=== Docker Arena RunPod Setup ==="

# 1. Install Dependencies
echo "[*] Installing System Prerequisites..."
apt-get update -qq
apt-get install -y -qq git curl python3 python3-pip

# 2. Install Docker if missing
if ! command -v docker &> /dev/null; then
    echo "[*] Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
else
    echo "[*] Docker already installed."
fi

# 3. Verify NVIDIA Docker Runtime
# RunPod usually has this, but we check just in case.
echo "[*] Checking GPU visibility..."
if docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "[+] GPU Docker is working!"
else
    echo "[!] WARNING: standard 'docker run --gpus all' failed. The inner containers might not see the GPU."
fi

echo "=== Setup Complete ==="
echo "You can now run: docker compose up -d"
