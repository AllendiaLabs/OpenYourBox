# Cloud Training Service (proprietary)

Phase 4 hosted backend for OpenYourBox cloud training. The VST remains the only
end-user training UI. This tree is **proprietary**; recipes invoke
`Backend/train_worker.py` so Gold `.pt` exports match Local Train.

Training is always real. There is no mock/fake advancement worker and no
`CLOUD_MOCK_WORKER` flag.

## Layout

- `api/` — HTTPS JSON control plane (auth, entitlement, jobs, artifacts, retention)
- `worker/` — real `train_runner` that materializes job packages and runs the
  shared `train_graph` recipe (`Backend/train_worker.py`). Legacy
  `train_steerable` packages are still accepted for existing tests.
- `storefront/` — WordPress customer + entitlement sync stubs (link/credits only;
  never fake train success)
- `tests/` — entitlement, concurrency, Stop, crash-fail, retention, and short
  real-train smoke tests
- `Dockerfile` — CUDA image for a RunPod GPU pod

## Staging environment

| Variable | Purpose | Default |
|----------|---------|---------|
| `CLOUD_API_HOST` | Bind address (`0.0.0.0` in Docker/RunPod) | `127.0.0.1` |
| `CLOUD_API_PORT` | Bind port | `8787` |
| `CLOUD_API_PUBLIC_URL` | Advertised API origin (signed download URLs, link bootstrap) | localhost, or RunPod HTTP proxy when `RUNPOD_POD_ID` is set |
| `CLOUD_STOREFRONT_URL` | Storefront origin used in `verification_url` | same as `CLOUD_API_PUBLIC_URL` |
| `CLOUD_DATA_DIR` | File-backed job/corpus/artifact store (persist across restarts for retention) | temp / in-memory; `/workspace/oyb-cloud-data` in the GPU image |
| `CLOUD_MOCK_ENTITLEMENT` | Staging ledger stub: `1` sufficient / `0` insufficient for new customers | `1` |
| `CLOUD_AUTO_WORKER` | `1` claims queued jobs and runs real training; `0` leaves jobs queued (tests) | `1` |
| `CLOUD_ALLOW_ANONYMOUS` | `1` allows Cloud jobs without Allendia sign-in (guest session) | `0` locally; `1` in the GPU image |
| `CLOUD_HEARTBEAT_TIMEOUT_SECONDS` | Stale worker heartbeat → `failed` / `worker_lost` | `120` |
| `CLOUD_DOWNLOAD_SECRET` | HMAC material for download tokens (never placed in query strings) | staging default |
| `RUNPOD_POD_ID` | Set by RunPod. Used only to derive the HTTPS proxy origin when `CLOUD_API_PUBLIC_URL` is unset | — |

In the plugin Train → Allendia account section, open **Cloud endpoint
(staging / RunPod)** and set:

- API base URL = `CLOUD_API_PUBLIC_URL` (example: `http://127.0.0.1:8787` or a
  RunPod `https://<POD_ID>-8787.proxy.runpod.net`)
- Storefront / link URL = `CLOUD_STOREFRONT_URL` (same origin for staging)

Apply endpoint, then Sign in. The same keys can still be edited in user-data
`cloud.xml` (`apiBaseUrlOverride`, `storefrontUrlOverride`).

Train controls are **Run** and **Stop** only (no Pause/Resume) for Local and Cloud.

With `CLOUD_ALLOW_ANONYMOUS=1` (default in the RunPod image), the plugin can Cloud
**Run** without Allendia sign-in. Sign-in remains available but optional.

## Run (staging)

```bash
pip install -r CloudService/requirements.txt
# plus existing Backend/train dependency install used for local Train
export CLOUD_API_PUBLIC_URL=http://127.0.0.1:8787
export CLOUD_DATA_DIR=/tmp/oyb-cloud-data
PYTHONPATH=. python -m uvicorn CloudService.api.app:app --host 127.0.0.1 --port 8787
```

Staging link completion (tests and the local link page): `POST /mock/link/complete`
with `{"user_code": "ABCD-EFGH", "customer_id": "cust-1"}`. This only verifies the
device-code account link; it does not mark training successful.

## RunPod GPU pod (real CUDA training)

The same API + `train_runner` image runs on a RunPod GPU pod. Training stays
asynchronous (submit → poll → download), so the RunPod HTTP proxy's 100-second
request timeout is compatible with job progress. Prefer the HTTPS proxy URL for
TLS. Large corpus uploads that cannot finish in ~100 seconds should use a TCP
port instead (no automatic TLS).

### Deploy

1. Runpod console → **Pods** → **Deploy**.
2. GPU: **RTX 4090** (or similar 24 GB) is enough for short `train_graph` jobs;
   pick more VRAM for long staged runs.
3. Template / container image: `YOUR_DOCKERHUB_USER/oyb-cloud-train:latest` (or your
   registry tag).
4. **Expose HTTP Ports**: `8787` (max one HTTP proxy port is typical).
5. Container disk ≥ 40 GB. Attach a network volume at `/workspace` if you want
   `CLOUD_DATA_DIR` to survive stop/terminate.
6. Environment:
   - `CLOUD_DOWNLOAD_SECRET` = a long random string (required before any real use)
   - `CLOUD_MOCK_ENTITLEMENT` = `1` for staging credits
   - `CLOUD_ALLOW_ANONYMOUS` = `1` if Cloud Run should work without Allendia sign-in
   - Optional: `CLOUD_API_PUBLIC_URL` if you are not using the HTTP proxy
7. Deploy on-demand. Wait until `GET /v1/health` returns `"ok": true` and
   `"cuda": true`.

Public origin (set this on the plugin):

```text
https://<POD_ID>-8787.proxy.runpod.net
```

Confirm:

```bash
curl -sS "https://<POD_ID>-8787.proxy.runpod.net/v1/health"
# {"ok": true, "cuda": true, "device": "cuda", "public_url": "https://..."}
```

CLI equivalent:

```bash
runpodctl pod create \
  --name oyb-cloud-train \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --image "YOUR_DOCKERHUB_USER/oyb-cloud-train:latest" \
  --container-disk-in-gb 40 \
  --volume-in-gb 50 \
  --ports "8787/http"
```

If your `runpodctl` build does not accept `--ports`, set **Expose HTTP Ports** to
`8787` in the console instead.

Stop the pod when idle so you are not billed for the GPU. Restarting or
replacing the pod allocates a new `POD_ID` (new proxy URL) and fails any
in-flight job (`worker_lost`). This slice does not auto-resume.

### Point the plugin at the pod

In Train → Allendia account → **Cloud endpoint (staging / RunPod)**:

- API base URL = `https://<POD_ID>-8787.proxy.runpod.net`
- Storefront / link URL = the same origin
- **Apply endpoint** (Allendia sign-in optional when `CLOUD_ALLOW_ANONYMOUS=1`)

(Or set the same keys in user-data `cloud.xml`.) Cloud **Run** then submits a
real GPU job.

## Tests

```bash
PYTHONPATH=. pytest CloudService/tests -q
```

## Security

- **Production requires TLS.** Staging HTTP on localhost is acceptable; do not
  skip TLS, entitlement, Bearer auth, or retention when promoting to a public host.
- Session tokens are `Authorization: Bearer` headers only (never query strings).
- Artifact bytes are served via time-limited signed path tokens under
  `/v1/downloads/{token}` — not raw job ids in public query strings.
- Do not log raw credentials, Bearer tokens, download secrets, or corpus contents.
- Entitlement and one-job-per-account gates stay enforced on staging.
- Corpus retention remains a 30-day sliding window from last use.
- RunPod HTTP proxy URLs are publicly reachable. Keep Bearer auth, a unique
  `CLOUD_DOWNLOAD_SECRET`, and entitlement gates enabled. The proxy supplies
  HTTPS; do not treat that as a reason to skip application auth.
