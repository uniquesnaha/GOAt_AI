# GOAt AI — GCP Console deployment runbook

Sequential record of the exact commands used to deploy via the GCP Console
(no `gcloud` CLI) on an L4 GPU VM, including the real issues hit along the
way and how they were fixed. Written after actually doing this once — treat
it as the source of truth over `deploy/gcloud/README.md`'s CLI-flavored
version if the two ever disagree.

## 0. Console setup (click-through, no terminal)

1. **Project + billing**: pick/create a project at console.cloud.google.com,
   confirm billing is linked.
2. **Enable Compute Engine API**: search "Compute Engine" → Enable.
3. **GPU quota**: IAM & Admin → Quotas → filter `NVIDIA_L4_GPUS` for your
   region → if `0`, select it → Edit Quotas → request an increase. This is
   a manual Google approval step; can take minutes to ~a day. Do this first.
4. **Firewall rule**: VPC network → Firewall → Create Firewall Rule
   - Name: `allow-http`
   - Target tags: `http-server`
   - Source range: `0.0.0.0/0`
   - Protocols/ports: TCP `80,443`
5. **Create the VM**: Compute Engine → VM instances → Create Instance
   - Name: `goat-ai-vm` (or similar)
   - Region/zone: one with L4 availability
   - Machine configuration → GPUs tab → NVIDIA L4 × 1 (auto-selects a G2
     machine series)
   - Boot disk → Change → OS: **Deep Learning on Linux**, a CUDA 12.1+
     image, 200GB disk
   - Firewall: check "Allow HTTP traffic" and "Allow HTTPS traffic"
   - Create.

   **Known trap**: if you pick a plain Ubuntu/Debian image instead of a
   Deep Learning VM image, `nvidia-smi`/`docker` won't exist and you'll need
   to install the NVIDIA driver manually or just recreate the VM with the
   right image — recreating is far less error-prone.

## 1. SSH in

VM instances list → click **SSH** next to the instance → opens a
browser-based terminal, no key setup needed.

## 2. Verify the base image

```bash
nvidia-smi          # should show your GPU (L4), driver version, CUDA version
python3 --version    # Deep Learning VM images ship an older bundled Python (3.10.x) — a bare "3.13.x" means wrong image
docker --version
docker compose version
```

If `docker`/`docker compose` are missing (they were, on this deploy, even
on the correct Deep Learning VM image variant — Docker isn't bundled in
every DL VM image flavor):

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER
newgrp docker   # or log out/back in via the SSH button

docker --version
docker compose version
```

## 3. NVIDIA Container Toolkit (so containers can see the GPU)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# real gate — confirms GPU is visible inside a container, not just on the host:
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## 4. Get the code (data is committed in the repo, no separate transfer)

```bash
git clone https://github.com/uniquesnaha/GOAt_AI.git
cd GOAt_AI
git rev-parse HEAD   # record this — it's your deployed commit
```

`data/` (chunks, embeddings, eval queries) is committed directly in this
repo (all files under GitHub's 100MB limit), so cloning is enough — no
separate `scp`/bucket step needed. `data/final_retrieval_tuning/` is the one
exception: only a `.gitkeep` placeholder is there. That file isn't needed to
serve queries or for the parity gate, only for the offline benchmark step —
drop it in before running that step.

Sanity check the clone actually pulled real data, not empty files:
```bash
ls -la data/embeddings_256/fixed_384_96/ta/    # embeddings.npy should be tens of MB, not 0 bytes
```

## 5. Secrets

```bash
cd deploy
cp .env.example .env
nano .env
```
Set `SARVAM_API_KEY=<your key>`, save (`Ctrl+O`, Enter, `Ctrl+X`).
Pasting into the browser SSH terminal: right-click → Paste, or
`Ctrl+Shift+V` / `Cmd+V`.

Verify: `cat .env` — no quotes around the value, no stray spaces.

## 6. Start Qdrant, build the backend image, GPU-check inside the container

```bash
docker compose --env-file .env up -d qdrant
docker compose --env-file .env build backend

docker compose --env-file .env run --rm backend \
  python -c "import torch, transformers, accelerate; print(torch.__version__, transformers.__version__, accelerate.__version__, torch.cuda.is_available())"
```
Must print versions + `True`, no traceback. Do not continue if it says `False`
or errors.

**Issues actually hit here and already fixed in the repo** (so a fresh
clone shouldn't hit these, but documenting in case a dependency shifts
again):
- `torch==2.4.0` pinned while `transformers` was left unbounded → pip
  installed a `transformers` release that hard-requires `torch>=2.5.0` and
  refused to load anything. Fixed by bumping to `torch==2.5.1` and
  `transformers>=4.51.0` (the latter also needed to be recent enough to
  recognize the Qwen3 architecture — downgrading transformers instead
  would have just traded one failure for "unrecognized model type: qwen3").
- `accelerate` was missing entirely — required by `transformers` for
  `device_map="cuda"`, which both the golden script and `engine.py` use.
  Added `accelerate>=0.33.0`.

## 7. Index Qdrant

```bash
docker compose --env-file .env run --rm backend python -m app.indexing.index_qdrant
```
Watch for `LOCAL QDRANT READY ✅` at the end. This only touches
`qdrant_client`/`numpy`/`pandas`, so it's unaffected by the torch/transformers
issues above even if you hit those in a different order.

Verify collections directly if you want to double check without re-running:
```bash
docker compose --env-file .env run --rm backend python - <<'PY'
from qdrant_client import QdrantClient
import os
c = QdrantClient(url=os.environ["QDRANT_URL"])
for name in ["hhgoa_fixed384_ta", "hhgoa_fixed384_hi"]:
    x = c.get_collection(name)
    print(name, x.points_count, x.indexed_vectors_count, x.status)
PY
```

## 8. Parity gate — must run on the HOST, not inside the backend container

**Why not in the container**: the golden script hardcodes
`QdrantClient(url="http://127.0.0.1:6333", ...)` — intentionally, since we
never edit that file. Inside the `backend` container, `127.0.0.1` means the
container's own loopback, not the separate `qdrant` container, so it fails
with `Connection refused`. Running on the host works because Qdrant's port
is published to the host's `127.0.0.1:6333` (see `deploy/docker-compose.yml`).

```bash
cd ~/GOAt_AI/backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # re-downloads torch etc. into the venv — expected, one-time

export GOAT_DATA_ROOT="$HOME/GOAt_AI/data"
export QDRANT_URL=http://127.0.0.1:6333
python -m parity.test_parity
```

**Issue hit**: `_ensure_golden_root_link()` tries to `mkdir` `/content`
(the golden script's hardcoded parent path) and fails with
`PermissionError: [Errno 13] Permission denied: '/content'` — a normal user
can't create directories at the filesystem root. One-time fix:
```bash
sudo mkdir -p /content
sudo chown $USER:$USER /content
```
Then re-run `python -m parity.test_parity`. (The script now prints this
exact fix if it hits that error again, instead of a raw traceback.)

Expect output ending in:
```
PARITY GATE: PASS
```
Every check (`fused parents`, `context string`, `context parent count`,
`prompt tokens`, `answer`) should read `OK` for every sample query, both
languages. If anything says `MISMATCH`, stop — that means packaging changed
RAG behavior and needs to be fixed, not the RAG retuned.

```bash
deactivate   # done with the venv
```

## 9. Bring the full stack up

```bash
cd ~/GOAt_AI/deploy
docker compose --env-file .env up -d --build
docker compose ps
```
Want `qdrant`, `backend`, `frontend` all `healthy`. `backend` may show
`starting` for a bit while `/readyz` waits on model load — the `hf_cache`
Docker volume means models downloaded during any earlier attempt are
already cached, so this is usually fast (well under the healthcheck's 5
minute `start_period`), not a fresh multi-GB download.

If it hangs or goes `unhealthy`:
```bash
docker compose logs --tail=100 backend
```

## 10. Verify

```bash
curl http://127.0.0.1:8000/readyz     # backend isn't published publicly, curl from the VM itself
curl -v http://127.0.0.1:80           # sanity check the frontend responds locally before testing externally
```

Find the external IP: Compute Engine → VM instances → the row either shows
an **External IP** column (scroll the table right if it's cut off), or
click the instance name to open its details page, where it's listed clearly
with a copy icon.

Then open `http://<VM_EXTERNAL_IP>` in a browser.

**Issue hit**: browser showed `ERR_CONNECTION_REFUSED` even though
`curl http://127.0.0.1:80` worked fine on the VM itself. "Refused" (not
"timed out") was the tell that this was a firewall/tagging problem, not a
container problem — GCP's VPC firewall silently drops disallowed traffic
(which shows as a timeout), so an active refusal from outside while it
works locally means the packet is reaching the VM's network layer but
nothing there is authorizing it externally.

Root cause: the VM either lacked the `http-server` network tag, or the
firewall rule allowing port 80 wasn't actually targeting whatever tag the
VM had (or had none). Fix, either of:
- Add the `http-server` tag to the VM (VM details page → Edit → Network
  tags → add `http-server` → save), matching whatever tag your firewall
  rule targets.
- Or edit the firewall rule (VPC network → Firewall → your rule → Edit) to
  target **All instances in the network** instead of a specific tag.

**Takeaway for next deploy**: when creating the VM, explicitly check
**"Allow HTTP traffic"** (and HTTPS) in the Firewall section of the Create
Instance page — this auto-applies the `http-server`/`https-server` tags at
creation time and avoids this whole detour.

## 11. Latency submission numbers

Requires `data/final_retrieval_tuning/candidate_ceiling_per_query.parquet`
(see step 4's note — drop this in first if you haven't).

```bash
docker compose --env-file .env exec backend \
  python -m benchmarks.benchmark_full_rag --per-language 50
curl http://127.0.0.1:8000/api/metrics
```

## Still outstanding (not yet done as of this runbook)

- **HTTPS**: browser mic (`getUserMedia`) requires a secure context.
  `frontend/nginx.conf` is plain HTTP only right now — needs a TLS
  terminator (e.g. Caddy in front of nginx) plus a domain name pointed at
  the VM's (ideally reserved static) external IP before the mic will work
  from a real browser, not just `curl`.
- **`final_retrieval_tuning/candidate_ceiling_per_query.parquet`**: still
  missing, only needed for the benchmark step, not for serving.
