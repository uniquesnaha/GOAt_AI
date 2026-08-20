#!/usr/bin/env bash
# GCP VM startup script: installs Docker + NVIDIA Container Toolkit and
# brings up the GOAt AI stack via docker-compose. Assumes the VM image
# already has the NVIDIA driver (e.g. deeplearning-platform-release images) —
# see deploy/gcloud/README.md if you're on a plain Ubuntu image instead.
set -euo pipefail

log() { echo "[goat-ai-startup] $*"; }

# --- Docker -----------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
fi

# --- NVIDIA Container Toolkit ------------------------------------------------
if ! docker info 2>/dev/null | grep -qi nvidia; then
  log "Installing NVIDIA Container Toolkit..."
  distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/"$distribution"/libnvidia-container.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

log "Verifying GPU is visible inside a container..."
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi || \
  log "WARNING: GPU not visible yet — check the NVIDIA driver install."

# --- Bring up the stack -------------------------------------------------------
# Expects the repo to already be at ~/GOAt_AI (see README.md step 4) and
# deploy/.env to be configured (step 5) before this runs, or re-run manually
# after those steps: `docker compose --env-file .env up -d --build`.
REPO_DIR="${HOME}/GOAt_AI"
if [ -f "${REPO_DIR}/deploy/docker-compose.yml" ] && [ -f "${REPO_DIR}/deploy/.env" ]; then
  log "Starting docker compose stack..."
  cd "${REPO_DIR}/deploy"
  docker compose --env-file .env up -d --build
else
  log "Repo/.env not found at ${REPO_DIR} yet — skipping compose up. Run it manually once code/data/.env are in place."
fi

log "Done."
