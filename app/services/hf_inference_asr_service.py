"""
ASR через Hugging Face Inference API (облако), без загрузки весов на машину.

Нужен HF_TOKEN. Аудио: webm → float32 @ 16 kHz → WAV (stdlib wave) для API.

Важно: для 汉字 нужны language=zh и task=transcribe в теле JSON внутри ключа "parameters".

InferenceClient передаёт extra_body как поля *рядом* с inputs — нельзя писать
extra_body={"parameters": {...}}, иначе получится parameters.parameters и язык не применится.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from app.services.hf_whisper_service import _load_audio_16k_mono, float32_mono_to_wav_bytes

logger = logging.getLogger(__name__)


def _whisper_inference_parameters(language_code: str) -> dict[str, Any]:
    """Параметры Whisper на HF Inference: язык вывода + задача транскрибации (не перевод)."""
    return {
        "language": language_code,
        "task": "transcribe",
    }


def transcribe_bytes_sync_inference(
    data: bytes,
    *,
    filename: str = "audio.webm",
    language: str | None = "zh",
) -> tuple[str, dict[str, Any]]:
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("Укажите HF_TOKEN в .env для Hugging Face Inference API")

    model_id = (os.getenv("HF_WHISPER_MODEL") or "openai/whisper-large-v3").strip()

    # Никогда не оставляем язык пустым: иначе Whisper выберет авто-язык (часто ru у русскоязычных).
    lang = (language or os.getenv("HF_ASR_LANGUAGE") or "zh").strip() or "zh"

    audio = _load_audio_16k_mono(data, filename)
    wav_bytes = float32_mono_to_wav_bytes(audio, sample_rate=16000)

    client = InferenceClient(token=token)

    params = _whisper_inference_parameters(lang)

    try:
        # extra_body → напрямую в JSON `parameters` (см. huggingface_hub HFInferenceBinaryInputTask).
        out = client.automatic_speech_recognition(
            wav_bytes,
            model=model_id,
            extra_body=params,
        )
    except HfHubHTTPError as e:
        # Не повторяем без language — это даёт неверный язык вывода.
        logger.exception("HF Inference ASR failed")
        raise RuntimeError(f"HF Inference API: {e}") from e

    text = ""
    if isinstance(out, dict):
        text = str(out.get("text", "") or "")
    else:
        text = str(getattr(out, "text", "") or "")

    text = text.strip()
    meta: dict[str, Any] = {
        "model": model_id,
        "backend": "huggingface_inference",
        "language": lang,
        "whisper_parameters": params,
    }
    return text, meta
