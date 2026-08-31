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
    CLOUD_AUTO_WORKER=1

WORKDIR /app

COPY CloudService/requirements.txt /tmp/cloud-requirements.txt
RUN pip install --no-cache-dir -r /tmp/cloud-requirements.txt \
    && pip install --no-cache-dir auraloss scipy \
    && rm /tmp/cloud-requirements.txt

COPY CloudService /app/CloudService
COPY Backend /app/Backend
COPY CloudService/runpod/entrypoint.sh /usr/local/bin/oyb-cloud-entrypoint
RUN chmod +x /usr/local/bin/oyb-cloud-entrypoint \
    && mkdir -p /workspace/oyb-cloud-data

EXPOSE 8787

CMD ["/usr/local/bin/oyb-cloud-entrypoint"]
