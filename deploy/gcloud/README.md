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

## 6. Index Qdrant, then run the parity gate

Do this once per fresh Qdrant volume, before serving traffic:

```bash
cd ~/GOAt_AI/backend
docker compose -f ../deploy/docker-compose.yml up -d qdrant
# from inside a Python env with backend/requirements.txt installed and GOAT_DATA_ROOT set:
python -m app.indexing.index_qdrant
python -m parity.test_parity
```

(Running the indexer/parity gate directly with a local Python env — rather
than inside the backend container — is usually faster to iterate on; the
container will pick up the already-populated Qdrant volume when it starts.)

## 7. Bring everything up

```bash
cd ~/GOAt_AI/deploy
docker compose --env-file .env up -d --build
```

Check readiness:

```bash
curl http://localhost:8000/readyz
```

The frontend is served on port 80. If you attached a static IP / DNS name,
point it there; otherwise use the VM's external IP.

## 8. Generate the latency submission numbers

```bash
cd ~/GOAt_AI/backend
python -m benchmarks.benchmark_full_rag --per-language 50
curl http://localhost:8000/api/metrics
```

This reproduces the same P50/P70/P90/P95/P100 report the golden script
produces, on 100+ deterministic queries, plus makes it available to the
frontend via `/api/metrics`.
