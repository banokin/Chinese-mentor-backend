"""
FastAPI entrypoint: CORS for Next.js, practice HTTP routes, pronunciation WebSocket.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from prometheus_client import make_asgi_app

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from app.agent_rag import routes as agent_routes
from app.pronunciation.routes import practice, websocket

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """HF token, LangSmith / Langfuse env."""
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if token and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    yield


app = FastAPI(
    title="Chinese Pronunciation Trainer API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(practice.router)
app.include_router(websocket.router)
app.include_router(agent_routes.router)

# Экспорт метрик для Prometheus.
app.mount("/metrics", make_asgi_app())
