#!/usr/bin/env bash
# Start the OpenYourBox cloud API + real GPU train worker.
# Used as the container CMD on RunPod (and any CUDA Docker host).
set -euo pipefail

HOST="${CLOUD_API_HOST:-0.0.0.0}"
PORT="${CLOUD_API_PORT:-8787}"
DATA_DIR="${CLOUD_DATA_DIR:-/workspace/oyb-cloud-data}"
mkdir -p "${DATA_DIR}"
export CLOUD_DATA_DIR="${DATA_DIR}"
export CLOUD_API_HOST="${HOST}"
export CLOUD_API_PORT="${PORT}"

if [ -z "${CLOUD_API_PUBLIC_URL:-}" ]; then
  if [ -n "${RUNPOD_POD_ID:-}" ]; then
    export CLOUD_API_PUBLIC_URL="https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net"
  else
    export CLOUD_API_PUBLIC_URL="http://127.0.0.1:${PORT}"
  fi
fi

if [ -z "${CLOUD_STOREFRONT_URL:-}" ]; then
  export CLOUD_STOREFRONT_URL="${CLOUD_API_PUBLIC_URL}"
fi

if [ "${CLOUD_DOWNLOAD_SECRET:-staging-dev-secret}" = "staging-dev-secret" ]; then
  echo "WARNING: CLOUD_DOWNLOAD_SECRET is the staging default. Set a unique secret on this host." >&2
fi

python - <<'PY'
import torch
print(
    "oyb-cloud-train device:",
    "cuda" if torch.cuda.is_available() else "cpu",
    flush=True,
)
if torch.cuda.is_available():
    print("oyb-cloud-train gpu:", torch.cuda.get_device_name(0), flush=True)
PY

exec python -m uvicorn CloudService.api.app:app --host "${HOST}" --port "${PORT}"
