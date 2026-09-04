# syntax=docker/dockerfile:1
# GPU image for the proprietary CloudService API + real train worker.
# Build from the repository root (needs Backend/ and CloudService/):
#   docker build --platform linux/amd64 -f CloudService/Dockerfile -t oyb-cloud-train .
#
# Do not pip-install Backend/requirements.txt here: it lists `torch` and would
# replace this image's CUDA wheel with a CPU build.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    CLOUD_API_HOST=0.0.0.0 \
    CLOUD_API_PORT=8787 \
    CLOUD_DATA_DIR=/workspace/oyb-cloud-data \
    CLOUD_AUTO_WORKER=1 \
    CLOUD_ALLOW_ANONYMOUS=1 \
    CLOUD_MOCK_ENTITLEMENT=1

WORKDIR /app

COPY CloudService/requirements.txt /tmp/cloud-requirements.txt
RUN pip install --no-cache-dir -r /tmp/cloud-requirements.txt \
    && pip install --no-cache-dir auraloss scipy \
    && rm /tmp/cloud-requirements.txt

COPY CloudService /app/CloudService
COPY Backend /app/Backend

# Container start script is baked into the image (no external shell file required).
RUN <<'EOF'
set -e
cat > /usr/local/bin/oyb-cloud-entrypoint <<'SCRIPT'
#!/usr/bin/env bash
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
SCRIPT
chmod +x /usr/local/bin/oyb-cloud-entrypoint
mkdir -p /workspace/oyb-cloud-data
EOF

EXPOSE 8787

CMD ["/usr/local/bin/oyb-cloud-entrypoint"]
