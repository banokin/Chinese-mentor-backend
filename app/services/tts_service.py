"""Озвучивание эталонной фразы через OpenAI Speech API (mp3)."""

from __future__ import annotations

import os

from openai import AsyncOpenAI

_openai_client: AsyncOpenAI | None = None


def _get_async_openai_client() -> AsyncOpenAI:
    """Create and cache AsyncOpenAI client inside TTS service."""
    global _openai_client
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


class TTSError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def synthesize_speech_mp3(
    text: str,
    *,
    model: str,
    voice: str,
) -> bytes:
    """
    Синтез речи в MP3. Нужен OPENAI_API_KEY.

    model: например tts-1 или gpt-4o-mini-tts
    voice: alloy, echo, fable, onyx, nova, shimmer (см. документацию OpenAI)
    """
    t = (text or "").strip()
    if not t:
        raise TTSError("empty_text", "Пустой текст для озвучки")
    if len(t) > 2000:
        raise TTSError("text_too_long", "Текст для озвучки слишком длинный")

    try:
        client = _get_async_openai_client()
    except ValueError as e:
        raise TTSError("no_openai_key", str(e)) from e

    try:
        resp = await client.audio.speech.create(
            model=model,
            voice=voice,  # type: ignore[arg-type]
            input=t,
            response_format="mp3",
        )
        data = await resp.aread()
    except Exception as e:
        raise TTSError("openai_tts_failed", str(e)) from e

    if not data:
        raise TTSError("empty_audio", "Пустой ответ TTS")
    return data
