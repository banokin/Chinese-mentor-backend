"""
FastAPI entrypoint: CORS for Next.js, practice HTTP routes, pronunciation WebSocket.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]


def _exit_if_project_venv_ignored() -> None:
    """
    При локальном backend/venv (или .venv) процесс должен идти из него.
    Иначе «uvicorn» из Conda тянет старый LangChain и падает на create_agent.
    """
    for name in ("venv", ".venv"):
        venv_dir = (_BACKEND_DIR / name).resolve()
        if not (venv_dir / "pyvenv.cfg").is_file():
            continue
        if os.path.normpath(sys.prefix) == os.path.normpath(str(venv_dir)):
            return
        if sys.platform == "win32":
            py = venv_dir / "Scripts" / "python.exe"
            cmd = f'"{py}" -m uvicorn app.main:app --reload'
        else:
            py = venv_dir / "bin" / "python"
            cmd = f"{py} -m uvicorn app.main:app --reload"
        print(
            "Найден каталог зависимостей "
            f"{name!r}, но интерпретатор другой (sys.prefix={sys.prefix!r}).\n"
            "Не вызывайте «голый» uvicorn из Conda. Из каталога backend выполните:\n"
            f"  {cmd}\n",
            file=sys.stderr,
        )
        raise SystemExit(2)


_exit_if_project_venv_ignored()

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

load_dotenv(_BACKEND_DIR / ".env")

from app.agent_rag.observability import log_observability_status
from app.agent_rag import routes as agent_routes
from app.pronunciation.routes import practice, websocket

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """HF token, LangSmith / Langfuse env."""
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if token and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    log_observability_status()
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
        "https://chinese-mentor-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(practice.router)
app.include_router(websocket.router)
app.include_router(agent_routes.router)
app.include_router(agent_routes.chat_router)

# Экспорт метрик для Prometheus.
app.mount("/metrics", make_asgi_app())
