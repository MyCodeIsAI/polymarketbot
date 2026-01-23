"""FastAPI application for PolymarketBot web dashboard.

Provides:
- REST API for bot management and data access
- WebSocket for real-time updates
- Static file serving for dashboard
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Global app instance
_app: Optional[FastAPI] = None

# Paths
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("web_dashboard_starting")

    # Initialize resources
    app.state.start_time = datetime.utcnow()
    app.state.ws_clients = set()

    yield

    # Cleanup
    logger.info("web_dashboard_stopping")


def create_app(
    title: str = "PolymarketBot Dashboard",
    debug: bool = False,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        title: Application title
        debug: Enable debug mode

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title=title,
        description="Web dashboard for PolymarketBot copy-trading system",
        version="1.0.0",
        docs_url="/api/docs" if debug else None,
        redoc_url="/api/redoc" if debug else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Setup templates
    templates = None
    if TEMPLATES_DIR.exists():
        templates = Jinja2Templates(directory=TEMPLATES_DIR)
        app.state.templates = templates

    # Include routers
    from .api.routes import router as api_router
    from .api.websocket import router as ws_router
    from ..discovery.routes import router as discovery_router

    app.include_router(api_router, prefix="/api")
    app.include_router(ws_router, prefix="/ws")
    app.include_router(discovery_router, prefix="/api")

    # Dashboard route
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Serve the main dashboard."""
        if templates:
            return templates.TemplateResponse(
                "dashboard.html",
                {"request": request, "title": title},
            )
        return HTMLResponse(
            content="<h1>Dashboard templates not found</h1>",
            status_code=500,
        )

    # Discovery page route
    @app.get("/discovery", response_class=HTMLResponse)
    async def discovery_page(request: Request):
        """Serve the account discovery page."""
        if templates:
            return templates.TemplateResponse(
                "discovery.html",
                {"request": request, "title": "Account Discovery"},
            )
        return HTMLResponse(
            content="<h1>Discovery templates not found</h1>",
            status_code=500,
        )

    # Infrastructure page route
    @app.get("/infrastructure", response_class=HTMLResponse)
    async def infrastructure_page(request: Request):
        """Serve the infrastructure configuration page."""
        if templates:
            return templates.TemplateResponse(
                "infrastructure.html",
                {"request": request, "title": "Infrastructure"},
            )
        return HTMLResponse(
            content="<h1>Infrastructure templates not found</h1>",
            status_code=500,
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
        }

    return app


def get_app() -> FastAPI:
    """Get or create the global app instance.

    Returns:
        FastAPI application
    """
    global _app
    if _app is None:
        _app = create_app()
    return _app
