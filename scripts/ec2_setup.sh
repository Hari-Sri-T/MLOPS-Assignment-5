#!/usr/bin/env bash
# One-time setup for a fresh EC2 instance (Ubuntu 22.04/24.04 or Amazon Linux 2023).
# Usage (on the instance):  bash ec2_setup.sh
set -euo pipefail

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y docker.io curl nginx
else
  sudo dnf install -y docker nginx
fi

sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

# 2 GB swap so TensorFlow fits comfortably on a 1 GB free-tier instance
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "Done. Log out and back in (or run 'newgrp docker') so docker works without sudo."
docker --version
free -h
