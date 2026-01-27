#!/usr/bin/env python3
"""
Standalone Discovery Server for Polymarket Account Discovery.

Runs the discovery web UI on configurable port (default: 8765).
Set DISCOVERY_PORT environment variable to override.

Access the discovery page at: http://localhost:{port}/discovery
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Configurable port via environment variable
DEFAULT_PORT = 8765
PORT = int(os.environ.get("DISCOVERY_PORT", DEFAULT_PORT))

# Cross-platform path handling
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


async def run_discovery_server():
    """Run the discovery web server."""
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.middleware.cors import CORSMiddleware

    # Create FastAPI app
    app = FastAPI(
        title="Polymarket Account Discovery",
        description="Discover profitable systematic traders on Polymarket",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Paths
    static_dir = PROJECT_ROOT / "src" / "web" / "static"
    templates_dir = PROJECT_ROOT / "src" / "web" / "templates"

    # Mount static files
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Setup templates
    templates = None
    if templates_dir.exists():
        templates = Jinja2Templates(directory=str(templates_dir))

    # Include discovery routes
    from src.discovery.routes import router as discovery_router
    app.include_router(discovery_router, prefix="/api")

    # Homepage route
    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """Serve the homepage."""
        if templates:
            index_path = templates_dir / "index.html"
            if index_path.exists():
                return templates.TemplateResponse(
                    "index.html",
                    {"request": request, "title": "Polymarket Tools"},
                )
        # Fallback: check for index.html directly
        index_path = templates_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text())
        # Default simple homepage
        return HTMLResponse(
            content="""<!DOCTYPE html>
<html><head><title>Polymarket Tools</title></head>
<body style="font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
<h1>Polymarket Tools</h1>
<ul>
<li><a href="/discovery">Account Discovery</a> - Find profitable systematic traders</li>
<li><a href="/infrastructure">Infrastructure</a> - System monitoring and configuration</li>
<li><a href="/docs">API Documentation</a></li>
</ul>
</body></html>""",
            status_code=200,
        )

    @app.get("/discovery", response_class=HTMLResponse)
    async def discovery_page(request: Request):
        """Serve the account discovery page."""
        if templates:
            return templates.TemplateResponse(
                "discovery.html",
                {"request": request, "title": "Account Discovery"},
            )
        # Fallback: read file directly
        template_path = templates_dir / "discovery.html"
        if template_path.exists():
            return HTMLResponse(content=template_path.read_text())
        return HTMLResponse(
            content="<h1>Discovery template not found</h1>",
            status_code=500,
        )

    @app.get("/infrastructure", response_class=HTMLResponse)
    async def infrastructure_page(request: Request):
        """Serve the infrastructure monitoring page."""
        if templates:
            return templates.TemplateResponse(
                "infrastructure.html",
                {"request": request, "title": "Infrastructure"},
            )
        # Fallback: read file directly
        template_path = templates_dir / "infrastructure.html"
        if template_path.exists():
            return HTMLResponse(content=template_path.read_text())
        return HTMLResponse(
            content="<h1>Infrastructure template not found</h1>",
            status_code=500,
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "discovery",
        }

    # Print startup message
    print("\n" + "=" * 60)
    print("  Polymarket Account Discovery Server")
    print("=" * 60)
    print(f"\n  Discovery URL: \033[1;36mhttp://localhost:{PORT}/discovery\033[0m")
    print(f"  API Docs: http://localhost:{PORT}/docs")
    print("\n  Press Ctrl+C to stop\n")
    print("-" * 60)

    # Run the server
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(run_discovery_server())
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
