from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database import init_db
from backend.api.routes_ab_testing import router as ab_testing_router
from backend.api.routes_clustering import router as clustering_router
from backend.api.routes_experiments import router as experiments_router
from backend.api.routes_clt import router as clt_router
from backend.api.routes_websocket import router as websocket_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_db()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Database initialization skipped: %s", exc
        )
    yield


app = FastAPI(
    title="A/B Testing & User Segmentation Engine",
    description="Enterprise-grade REST API for statistical A/B testing, "
                "Bayesian inference, and user persona clustering.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ab_testing_router, prefix="/api/v1", tags=["A/B Testing"])
app.include_router(clustering_router, prefix="/api/v1", tags=["Clustering"])
app.include_router(experiments_router, prefix="/api/v1", tags=["Experiments"])
app.include_router(clt_router, prefix="/api/v1", tags=["CLT"])
app.include_router(websocket_router, tags=["WebSocket"])


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "A/B Testing Engine API",
        "version": "2.0.0",
        "status": "healthy",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
