# Deploying GOAt AI to a GCP GPU VM

This deploys `deploy/docker-compose.yml` (Qdrant + backend + frontend) to a
single GPU VM (L4 or T4). The RAG engine hardcodes `device="cuda"` on
purpose (see the plan/README) — this must be a GPU instance, there is no
CPU fallback.

## 1. Create the VM

L4 example (cheaper, widely available; swap `nvidia-l4` for `nvidia-tesla-t4`
if you need T4 instead):

```bash
gcloud compute instances create goat-ai-vm \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu121 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB \
  --metadata-from-file=startup-script=startup-script.sh \
  --tags=http-server,https-server
```

The `deeplearning-platform-release` image family ships the NVIDIA driver
pre-installed, which avoids the most common source of GPU-VM setup failures.
If you use a plain Ubuntu image instead, install the driver manually first
(`sudo apt install nvidia-driver-535` or later, then reboot) before Docker
containers can see the GPU.

## 2. Open firewall for the frontend

```bash
gcloud compute firewall-rules create allow-http \
  --allow=tcp:80,tcp:443 \
  --target-tags=http-server,https-server
```

## 3. Install Docker + NVIDIA Container Toolkit

The `startup-script.sh` in this folder does this automatically on first
boot. To run it manually (or re-run after a driver update):

```bash
gcloud compute ssh goat-ai-vm --zone=us-central1-a
sudo bash startup-script.sh
```

Verify GPU access from inside a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## 4. Get the code and data onto the VM

```bash
gcloud compute scp --recurse ../.. goat-ai-vm:~/GOAt_AI --zone=us-central1-a
```

Then drop your data files into `~/GOAt_AI/data/` per **DATA_SETUP.md** —
either `scp` them directly or copy from wherever you generated them
(the same embeddings/chunks the golden benchmark script used).

## 5. Configure secrets

```bash
cp deploy/.env.example deploy/.env
# edit deploy/.env and set SARVAM_API_KEY
```

## 6. Build images, then index Qdrant, then run the parity gate

```bash
cd ~/GOAt_AI/deploy
docker compose --env-file .env up -d qdrant
docker compose --env-file .env build backend

# sanity check the GPU is actually visible to the container first:
docker compose --env-file .env run --rm backend \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# must print: True Tesla T4  (or your GPU's name) — do not continue if it says False

docker compose --env-file .env run --rm backend python -m app.indexing.index_qdrant
docker compose --env-file .env run --rm backend python -m parity.test_parity
```

`backend`'s container mounts `../scripts` read-only, so the parity gate can
load the untouched golden script from inside the container and compare it
against `app.rag.engine` — same dense/BM25/fused parent IDs, same context
string, same prompt tokens, same generated answer. If it fails, that's a
packaging bug to fix, not a reason to retune the RAG.

Qdrant's ports are published to `127.0.0.1` only (not the public interface),
so they're reachable from this VM but never from the internet — this is
true defense in depth, independent of your firewall rules.

## 7. Bring everything up

```bash
cd ~/GOAt_AI/deploy
docker compose --env-file .env up -d --build
docker compose ps
```

`qdrant`, `backend`, and `frontend` should all show `healthy` — backend's
healthcheck hits `/readyz` (can take a few minutes on first boot while the
Qwen models download), frontend's depends on backend being healthy first.

```bash
curl http://127.0.0.1:8000/readyz   # from the VM; backend isn't published publicly
```

The frontend is served on port 80 (the only public surface — backend/Qdrant
ports are loopback-only). If you attached a static IP / DNS name, point it
there; otherwise use the VM's external IP.

## 8. Generate the latency submission numbers

```bash
cd ~/GOAt_AI/deploy
docker compose --env-file .env exec backend \
  python -m benchmarks.benchmark_full_rag --per-language 50
curl http://127.0.0.1:8000/api/metrics
```

This reproduces the same P50/P70/P90/P95/P100 report the golden script
produces, on 100+ deterministic queries, plus makes it available to the
frontend via `/api/metrics`.
