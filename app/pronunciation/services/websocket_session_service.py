"""
In-memory WebSocket session: buffers audio chunks, optional partial ASR, final on stop.

MVP: no real streaming ASR — we periodically send the whole buffer to the same ASR service.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field

from app.services.asr_service import ASRError, transcribe_bytes
from app.pronunciation.services.scoring_service import evaluate_expected_vs_recognized
from app.pronunciation.services.transcription_match_service import evaluate_text_match

logger = logging.getLogger(__name__)


@dataclass
class PronunciationSessionState:
    """
    One session per WebSocket connection (stateless between connections).

    Junior note: all bytes live in RAM only — long recordings will use more memory.
    """

    expected_text: str = ""
    audio_buffer: bytearray = field(default_factory=bytearray)
    chunks_since_partial: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Transcribe every N audio_chunk messages for "live" feel (MVP).
    partial_every_n_chunks: int = 4


async def handle_start(state: PronunciationSessionState, expected_text: str) -> None:
    """Reset buffer and store target phrase."""
    async with state.lock:
        state.expected_text = (expected_text or "").strip()
        state.audio_buffer.clear()
        state.chunks_since_partial = 0


async def append_chunk(state: PronunciationSessionState, chunk_b64: str) -> bytes | None:
    """
    Decode base64 chunk and append. Returns copy of current buffer for optional partial ASR outside lock.

    Caller should run partial transcription without holding the lock for long.
    """
    async with state.lock:
        try:
            raw = base64.b64decode(chunk_b64, validate=True)
        except Exception:
            raise ValueError("Некорректный base64 в audio_chunk") from None
        if not raw:
            return None
        state.audio_buffer.extend(raw)
        state.chunks_since_partial += 1
        if state.chunks_since_partial >= state.partial_every_n_chunks:
            state.chunks_since_partial = 0
            return bytes(state.audio_buffer)
        return None


async def snapshot_buffer(state: PronunciationSessionState) -> tuple[str, bytes]:
    """Return expected text and full audio bytes."""
    async with state.lock:
        return state.expected_text, bytes(state.audio_buffer)


async def run_partial_transcription(buffer: bytes) -> tuple[str, dict]:
    """Best-effort partial; failures are logged and surfaced as soft errors."""
    if not buffer:
        return "", {}
    try:
        text, meta = await transcribe_bytes(buffer, filename="stream.webm")
        return text, meta
    except ASRError as e:
        logger.info("Partial ASR skipped: %s", e.code)
        raise


async def build_final_result(expected_text: str, recognized_text: str) -> dict:
    """Shape final_result.data like HTTP evaluate (scores + feedback)."""
    feedback, scores = evaluate_expected_vs_recognized(expected_text, recognized_text)
    match_ok, match_msg = evaluate_text_match(expected_text, recognized_text)
    hanzi = recognized_text.strip()
    return {
        "recognized_text": recognized_text,
        "transcription_hanzi": hanzi,
        "text_matches_expected": match_ok,
        "text_match_message_ru": match_msg,
        "expected_text": expected_text,
        "feedback": [f.model_dump() for f in feedback],
        "scores": scores.model_dump(),
    }
