from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import routers, database, and config
from routers import detect, stats
from database.db import init_db
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create directories and initialize database
    settings.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    yield
    # Shutdown: (cleanup if needed)


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

# Mount static files for images
app.mount("/images", StaticFiles(directory="images"), name="images")

# Apply Routers
app.include_router(detect.router)
app.include_router(stats.router)


@app.get("/")
async def root():
    return {"message": f"{settings.API_TITLE} is running. Check /docs for API documentation."}
