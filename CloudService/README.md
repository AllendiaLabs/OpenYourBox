# Cloud Training Service (proprietary)

Phase 4 hosted backend for OpenYourBox cloud training. The VST remains the only
end-user training UI. This tree is **proprietary**; recipes conceptually match
`Backend/train_worker.py` for Gold `.pt` parity.

## Layout

- `api/` — HTTPS JSON control plane (auth, entitlement, jobs, artifacts, retention)
- `worker/` — mock/CPU worker plus GPU runner that reuses local train recipes
- `storefront/` — WordPress customer + entitlement sync stubs (server-side)
- `tests/` — entitlement, concurrency, control, and retention contract tests

## Staging environment

| Variable | Purpose | Default |
|----------|---------|---------|
| `CLOUD_API_HOST` | Bind address | `127.0.0.1` |
| `CLOUD_API_PORT` | Bind port | `8787` |
| `CLOUD_API_PUBLIC_URL` | Advertised API origin (signed URLs, link bootstrap) | `http://127.0.0.1:8787` |
| `CLOUD_STOREFRONT_URL` | Storefront origin used in `verification_url` | `http://127.0.0.1:8787` |
| `CLOUD_MOCK_ENTITLEMENT` | `1` sufficient / `0` insufficient for new customers | `1` |
| `CLOUD_MOCK_WORKER` | `1` advances queued jobs in-process | `1` |
| `CLOUD_DATA_DIR` | File-backed job/corpus/artifact store | temp dir |

The VST no longer exposes API/storefront URL fields. For local staging, set
overrides in the plugin `cloud.xml` (`apiBaseUrlOverride`, `storefrontUrlOverride`)
to `CLOUD_API_PUBLIC_URL` / `CLOUD_STOREFRONT_URL` (same origin for the mock link
page), or temporarily change the product defaults in `GraphTypes.h`.

## Run (dev)

```bash
pip install -r CloudService/requirements.txt
PYTHONPATH=. python -m uvicorn CloudService.api.app:app --host 127.0.0.1 --port 8787
```

Mock link completion (also used by tests): `POST /mock/link/complete` with
`{"user_code": "ABCD-EFGH", "customer_id": "cust-1"}`.

## Tests

```bash
PYTHONPATH=. pytest CloudService/tests -q
```

## Security

Production requires TLS. Session tokens are Bearer headers only (never query
strings). Artifact bytes are served via time-limited signed download paths.
Do not log raw credentials or corpus contents.
