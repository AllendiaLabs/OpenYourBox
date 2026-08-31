"""ASGI application wiring for the proprietary cloud training control plane."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from CloudService.api.artifacts import router as artifacts_router
from CloudService.api.auth import CloudAPIError, cloud_api_error_handler, router as auth_router
from CloudService.api.auth_link import router as auth_link_router
from CloudService.api.entitlement_routes import router as entitlement_router
from CloudService.api.jobs import (
    auto_worker_enabled,
    router as jobs_router,
    start_supervisor,
    stop_supervisor,
)
from CloudService.api.runtime import (
    accelerator_health,
    apply_runtime_defaults,
    resolve_public_api_url,
)
from CloudService.storefront.link_mock import router as storefront_link_router

health_router = APIRouter()


@health_router.get("/v1/health")
def health() -> dict[str, Any]:
    """Unauthenticated liveness probe, including CUDA visibility on GPU hosts."""
    payload: dict[str, Any] = {"ok": True, **accelerator_health()}
    payload["public_url"] = resolve_public_api_url()
    return payload


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start the real-worker supervisor for staging/production processes."""
    apply_runtime_defaults()
    if auto_worker_enabled():
        start_supervisor()
    try:
        yield
    finally:
        stop_supervisor()


def create_app() -> FastAPI:
    """Build the FastAPI application with all control-plane routers."""
    application = FastAPI(title="OpenYourBox Cloud Training", lifespan=lifespan)
    application.add_exception_handler(CloudAPIError, cloud_api_error_handler)
    application.include_router(health_router)
    application.include_router(auth_router)
    application.include_router(auth_link_router)
    application.include_router(entitlement_router)
    application.include_router(jobs_router)
    application.include_router(artifacts_router)
    application.include_router(storefront_link_router)

    @application.get("/")
    def root() -> JSONResponse:
        return JSONResponse(
            {"service": "openyourbox-cloud", "public_url": resolve_public_api_url()}
        )

    return application


app = create_app()
