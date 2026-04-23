"""Pydantic models for practice API and WebSocket payloads."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TranscriptionMeta(BaseModel):
    """Optional metadata from ASR (kept extensible)."""

    model: str | None = None
    duration_hint_sec: float | None = None


class TranscriptionResponse(BaseModel):
    """Stable JSON shape for HTTP transcription endpoint."""

    recognized_text: str
    meta: dict[str, Any] = Field(default_factory=dict)


class FeedbackItem(BaseModel):
    """One scored difference for the learner (Russian UI)."""

    index: int
    type: Literal[
        "correct",
        "tone_error",
        "syllable_error",
        "missing_syllable",
        "extra_syllable",
    ]
    expected: str | None = None
    actual: str | None = None
    message_ru: str


class FinalScores(BaseModel):
    """Aggregate metrics after syllable+tone alignment."""

    accuracy: float = Field(ge=0.0, le=1.0)
    tone_accuracy: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    fluency: float = Field(ge=0.0, le=1.0)


class PracticeEvaluateResponse(BaseModel):
    """Full practice round result."""

    recognized_text: str
    # Отдельное поле под UI: иероглифы, которые вернул ASR (обычно совпадает с recognized_text).
    transcription_hanzi: str = Field(description="Распознанная моделью фраза (汉字) для отображения.")
    text_matches_expected: bool = Field(
        description="True, если после нормализации совпадает с ожидаемой фразой.",
    )
    text_match_message_ru: str = Field(description="Короткое пояснение по-русски про совпадение текста.")
    feedback: list[FeedbackItem]
    scores: FinalScores


class ErrorDetail(BaseModel):
    """Structured API error (no stack traces to client)."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
