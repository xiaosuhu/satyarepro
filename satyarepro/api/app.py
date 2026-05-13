from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routers import audit, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="SatyaRepro",
        description="Agentic AI tool for biomedical reproducibility auditing",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(audit.router)
    return app


app = create_app()
