# CI/CD 테스트
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Import routers, database, and config
from routers import detect, stats, sessions, images, monitoring, feedback
from database.db import init_db
from database import db
from config import settings

# 로거 설정
logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create directories and initialize database
    settings.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    logger.info("FastAPI application started")
    logger.info(f"Slack alert enabled: {settings.SLACK_ALERT_ENABLED}")

    yield

    # Shutdown
    await db.close_db()
    logger.info("FastAPI application stopped")


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Static files mount removed (S3 Migration)
# app.mount("/images", StaticFiles(directory="images"), name="images")
# Static files mount removed (S3 Migration)
# app.mount("/images", StaticFiles(directory="images"), name="images")

# Apply Routers
app.include_router(detect.router)
app.include_router(stats.router)
app.include_router(sessions.router)
app.include_router(images.router)
app.include_router(monitoring.router)
app.include_router(feedback.router)


@app.get("/")
async def root():
    return {"message": f"{settings.API_TITLE} is running. Check /docs for API documentation."}
