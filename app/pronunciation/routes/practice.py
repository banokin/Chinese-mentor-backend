"""Thin HTTP routes for practice round (audio upload + evaluation)."""

from __future__ import annotations

import logging
import os
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from app.pronunciation.schemas import ErrorDetail, PracticeEvaluateResponse, TranscriptionResponse
from app.services.asr_service import ASRError, transcribe_bytes
from app.pronunciation.services.scoring_service import evaluate_expected_vs_recognized
from app.pronunciation.services.transcription_match_service import evaluate_text_match
from app.services.tts_service import TTSError, synthesize_speech_mp3

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/practice", tags=["practice"])


def _as_http(exc: ASRError) -> HTTPException:
    return HTTPException(
        status_code=400 if exc.code in {"empty_audio", "invalid_format"} else 502,
        detail=ErrorDetail(code=exc.code, message=exc.message, details=exc.details).model_dump(),
    )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_only(
    audio: Annotated[UploadFile, File(description="Recorded audio")],
) -> TranscriptionResponse:
    """ASR only — returns recognized_text + meta (OpenAI)."""
    data = await audio.read()
    try:
        text, meta = await transcribe_bytes(data, filename=audio.filename or "audio.webm")
    except ASRError as e:
        raise _as_http(e) from e
    return TranscriptionResponse(recognized_text=text, meta=meta)


@router.post("/example-speech")
async def example_speech(
    text: Annotated[str, Form()],
) -> Response:
    """
    MP3 с озвучкой эталонной фразы (OpenAI TTS). Без OPENAI_API_KEY — 503.
    """
    try:
        audio = await synthesize_speech_mp3(
            text,
            model=(os.getenv("OPENAI_TTS_MODEL") or "tts-1").strip() or "tts-1",
            voice=(os.getenv("OPENAI_TTS_VOICE") or "nova").strip() or "nova",
        )
    except TTSError as e:
        status = 400 if e.code in {"empty_text", "text_too_long"} else 503
        raise HTTPException(
            status_code=status,
            detail=ErrorDetail(code=e.code, message=e.message, details={}).model_dump(),
        ) from e

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": 'inline; filename="example.mp3"',
            "Cache-Control": "no-store",
        },
    )


@router.post("/evaluate", response_model=PracticeEvaluateResponse)
async def evaluate_pronunciation(
    expected_text: Annotated[str, Form()],
    audio: Annotated[UploadFile, File()],
) -> PracticeEvaluateResponse:
    """Transcribe then syllable+tone scoring against expected Chinese phrase."""
    data = await audio.read()
    try:
        recognized, _meta = await transcribe_bytes(data, filename=audio.filename or "audio.webm")
    except ASRError as e:
        raise _as_http(e) from e

    feedback, scores = evaluate_expected_vs_recognized(expected_text, recognized)
    match_ok, match_msg = evaluate_text_match(expected_text, recognized)
    hanzi = recognized.strip()
    return PracticeEvaluateResponse(
        recognized_text=recognized,
        transcription_hanzi=hanzi,
        text_matches_expected=match_ok,
        text_match_message_ru=match_msg,
        feedback=feedback,
        scores=scores,
    )
