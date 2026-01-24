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
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    logger.info("web_dashboard_starting")

    # Load environment variables
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Initialize resources
    app.state.start_time = datetime.utcnow()
    app.state.ws_clients = set()

    # Auto-start scanner if AUTO_START_SCANNER env var is set (or default to True)
    auto_start = os.getenv("AUTO_START_SCANNER", "true").lower() in ("true", "1", "yes")

    if auto_start:
        # Start scanner in background after a short delay
        asyncio.create_task(_auto_start_scanner())

    yield

    # Cleanup - stop scanner
    await _cleanup_scanner()
    logger.info("web_dashboard_stopping")


async def _auto_start_scanner():
    """Auto-start the scanner service after app initialization."""
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from contextlib import contextmanager

    # Wait for app to fully initialize
    await asyncio.sleep(2)

    try:
        from .api.insider_routes import _get_or_create_scanner
        from .api.dependencies import get_db

        # Load .env for API keys
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        # Get database manager and create session factory
        db_manager = get_db()

        @contextmanager
        def session_factory():
            with db_manager.session() as session:
                yield session

        scanner = await _get_or_create_scanner(session_factory)

        if not scanner.is_running:
            from ..insider_scanner.scanner_service import ScannerMode
            mode = os.getenv("SCANNER_MODE", "full").lower()
            scanner_mode = ScannerMode.FULL if mode == "full" else (
                ScannerMode.SYBIL_ONLY if mode == "sybil" else ScannerMode.INSIDER_ONLY
            )

            success = await scanner.start(mode=scanner_mode)
            if success:
                logger.info("scanner_auto_started", mode=scanner_mode.value)
            else:
                logger.warning("scanner_auto_start_failed")

    except Exception as e:
        logger.error("scanner_auto_start_error", error=str(e))


async def _cleanup_scanner():
    """Stop scanner on app shutdown."""
    try:
        from .api.insider_routes import _scanner_service

        if _scanner_service and _scanner_service.is_running:
            await _scanner_service.stop()
            logger.info("scanner_stopped_on_shutdown")

    except Exception as e:
        logger.error("scanner_cleanup_error", error=str(e))


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
    from .api.insider_routes import router as insider_router
    from ..discovery.routes import router as discovery_router

    app.include_router(api_router, prefix="/api")
    app.include_router(ws_router, prefix="/ws")
    app.include_router(insider_router, prefix="/api")
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

    # Scanner page route
    @app.get("/scanner", response_class=HTMLResponse)
    async def scanner_page(request: Request):
        """Serve the scanner dashboard."""
        if templates:
            return templates.TemplateResponse(
                "scanner.html",
                {"request": request, "title": "Scanner"},
            )
        return HTMLResponse(
            content="<h1>Scanner templates not found</h1>",
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


# Create app instance for uvicorn
app = get_app()
