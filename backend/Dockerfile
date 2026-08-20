# CUDA 12.1 + cuDNN runtime. torch's PyPI wheel bundles its own CUDA runtime
# libs (nvidia-*-cu12 packages) independent of this base image's toolkit
# version, so this only needs to be new enough for the driver's compat
# range — confirmed working against driver 580.x / CUDA 13.0 on an L4.
# Works on GCP L4/T4 GPU VMs with a recent NVIDIA driver + nvidia-container-toolkit.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip python3.11-venv curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app/backend

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

# Hugging Face model cache — mount a volume here in docker-compose so the
# embedder/generator aren't re-downloaded on every container restart.
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
