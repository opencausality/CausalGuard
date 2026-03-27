"""FastAPI application factory for CausalGuard.

The REST API is optional — the primary interface is the CLI.  Start it with:

    causalguard serve --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    from causalguard import __version__
    from causalguard.api.routes import router

    app = FastAPI(
        title="CausalGuard API",
        description=(
            "Build and verify causal safety cases for AI systems. "
            "Every hazard→harm path is formally checked against your mitigations."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app
