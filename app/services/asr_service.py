"""
Speech-to-text: OpenAI API, локальный Whisper или HF Inference API (см. ASR_BACKEND).

TODO(streaming): частичная транскрипция при стабильном chunked API / буфере.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from openai import APIError, APITimeoutError, BadRequestError, RateLimitError

logger = logging.getLogger(__name__)
_openai_client: AsyncOpenAI | None = None

# Разумный лимит для MVP (при необходимости увеличьте).
_MAX_BYTES = 25 * 1024 * 1024
_ALLOWED_SUFFIXES = (".webm", ".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg")


class ASRError(Exception):
    """Доменная ошибка с кодом для HTTP/WebSocket."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _get_async_openai_client() -> AsyncOpenAI:
    """Create and cache AsyncOpenAI client inside ASR service."""
    global _openai_client
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


def _validate_audio_payload(data: bytes, filename: str) -> None:
    if not data:
        raise ASRError("empty_audio", "Пустой аудиофайл", {"size": 0})
    if len(data) > _MAX_BYTES:
        raise ASRError(
            "audio_too_large",
            "Файл слишком большой",
            {"size": len(data), "max": _MAX_BYTES},
        )
    lower = filename.lower()
    if not any(lower.endswith(s) for s in _ALLOWED_SUFFIXES):
        raise ASRError(
            "invalid_format",
            "Неподдерживаемый формат аудио",
            {"filename": filename, "allowed": list(_ALLOWED_SUFFIXES)},
        )


async def _transcribe_openai(
    data: bytes,
    *,
    filename: str,
    language: str | None,
) -> tuple[str, dict[str, Any]]:
    """Транскрипция через OpenAI Audio API."""
    model = (os.getenv("OPENAI_TRANSCRIPTION_MODEL") or "whisper-1").strip() or "whisper-1"

    try:
        client = _get_async_openai_client()
    except ValueError as e:
        raise ASRError("config", str(e), {}) from e

    buffer = io.BytesIO(data)
    buffer.name = filename

    try:
        tr = await client.audio.transcriptions.create(
            model=model,
            file=buffer,
            language=language,
        )
    except BadRequestError as e:
        logger.warning("OpenAI BadRequest: %s", e)
        raise ASRError(
            "openai_bad_request",
            "Запрос к распознаванию отклонён",
            {"body": getattr(e, "body", None)},
        ) from e
    except (APITimeoutError, RateLimitError) as e:
        logger.warning("OpenAI transient: %s", e)
        raise ASRError(
            "openai_transient",
            "Временная ошибка сервиса распознавания",
            {"type": type(e).__name__},
        ) from e
    except APIError as e:
        logger.exception("OpenAI APIError")
        raise ASRError(
            "openai_failure",
            "Ошибка OpenAI при распознавании",
            {"type": type(e).__name__},
        ) from e
    except Exception as e:
        logger.exception("Unexpected ASR failure (OpenAI)")
        raise ASRError(
            "openai_failure",
            "Не удалось выполнить распознавание",
            {"type": type(e).__name__},
        ) from e

    text = (getattr(tr, "text", None) or "").strip()
    meta: dict[str, Any] = {"model": model, "backend": "openai"}
    return text, meta


async def transcribe_bytes(
    data: bytes,
    *,
    filename: str = "audio.webm",
    language: str | None = "zh",
) -> tuple[str, dict[str, Any]]:
    """
    Транскрипция сырых байтов: выбор бэкенда по ASR_BACKEND в .env.

    Returns (recognized_text, meta dict).
    """
    _validate_audio_payload(data, filename)
    asr_backend = (os.getenv("ASR_BACKEND") or "huggingface_inference").strip()

    if asr_backend == "huggingface_inference":
        try:
            from app.services.hf_inference_asr_service import transcribe_bytes_sync_inference

            return await asyncio.to_thread(
                transcribe_bytes_sync_inference,
                data,
                filename=filename,
                language=language,
            )
        except Exception as e:
            logger.exception("HF Inference ASR failure")
            msg = str(e)
            code = "hf_inference_failure"
            lower = msg.lower()
            if "декодировать аудио" in msg or "ffmpeg" in lower or "декодирован" in lower:
                code = "audio_decode"
            if "hf_token" in lower or "401" in msg or "403" in msg:
                code = "hf_auth"
            raise ASRError(
                code,
                "Ошибка ASR через Hugging Face Inference API. Проверьте HF_TOKEN, квоту и HF_WHISPER_MODEL.",
                {"type": type(e).__name__, "detail": msg[:1200]},
            ) from e

    if asr_backend == "huggingface":
        try:
            from app.services.hf_whisper_service import transcribe_bytes_sync

            return await asyncio.to_thread(
                transcribe_bytes_sync,
                data,
                filename=filename,
                language=language,
            )
        except Exception as e:
            logger.exception("HF Whisper ASR failure")
            msg = str(e)
            code = "hf_asr_failure"
            lower = msg.lower()
            if "декодировать аудио" in msg or "ffmpeg" in lower or "декодирован" in lower:
                code = "audio_decode"
            if "out of memory" in lower or "mps" in lower:
                code = "hf_oom"
            raise ASRError(
                code,
                "Ошибка локального ASR (HF). Если audio_decode — см. details.detail (декодирование webm). Иначе RAM/GPU и HF_TOKEN.",
                {"type": type(e).__name__, "detail": msg[:1200]},
            ) from e

    # openai
    return await _transcribe_openai(data, filename=filename, language=language)
