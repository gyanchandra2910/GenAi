"""
main.py — Application Entry Point
===================================
AI for the Indian Investor (NSE Intelligence Platform)

Bootstraps the FastAPI application, configures CORS, registers all API routers,
and exposes a health-check endpoint for infrastructure monitoring.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan (startup / shutdown hooks) ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and graceful shutdown."""
    logger.info("🚀  AI for the Indian Investor — starting up …")
    logger.info("📡  Environment : %s", settings.APP_ENV)
    yield
    logger.info("🛑  Shutting down gracefully …")


# ─── Application Factory ──────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application instance.

    Returns
    -------
    FastAPI
        Fully configured application ready to be served by Uvicorn.
    """
    app = FastAPI(
        title="AI for the Indian Investor",
        description=(
            "NSE Intelligence Platform — turns raw financial data into "
            "actionable investment insights using AI agents."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    from api.v1.router import api_v1_router  # noqa: PLC0415
    app.include_router(api_v1_router, prefix="/api/v1")

    return app


# ─── Instantiate ──────────────────────────────────────────────────────────────
app: FastAPI = create_app()


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get(
    "/health",
    tags=["Infrastructure"],
    summary="Health Check",
    response_description="Service liveness and basic runtime metadata.",
)
async def health_check() -> JSONResponse:
    """
    Liveness probe for load-balancers and orchestration systems (e.g. ECS / k8s).

    Returns a 200 OK with a JSON payload confirming the service is alive,
    the current environment, and a server-side epoch timestamp.
    """
    return JSONResponse(
        content={
            "status": "ok",
            "service": "AI for the Indian Investor",
            "environment": settings.APP_ENV,
            "timestamp_utc": time.time(),
        }
    )


# ─── Dev Entrypoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.APP_ENV == "development",
        log_level="info",
    )
