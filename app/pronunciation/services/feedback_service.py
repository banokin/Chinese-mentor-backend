"""
Russian short explanations for scoring steps (learner-facing copy).
"""

from __future__ import annotations

from app.pronunciation.schemas import FeedbackItem
from app.pronunciation.services.alignment import AlignmentStep


def _msg_for_step(step: AlignmentStep, seq_index: int) -> FeedbackItem:
    """Build one FeedbackItem with a stable Russian message."""
    op = step.op
    exp = step.expected
    act = step.actual

    if op == "correct":
        return FeedbackItem(
            index=seq_index,
            type="correct",
            expected=exp,
            actual=act,
            message_ru="Верно",
        )
    if op == "tone_error":
        return FeedbackItem(
            index=seq_index,
            type="tone_error",
            expected=exp,
            actual=act,
            message_ru=f"Неверный тон в слоге '{exp}'",
        )
    if op == "syllable_error":
        return FeedbackItem(
            index=seq_index,
            type="syllable_error",
            expected=exp,
            actual=act,
            message_ru=f"Ожидалось '{exp}', получено '{act}'",
        )
    if op == "missing_syllable":
        return FeedbackItem(
            index=seq_index,
            type="missing_syllable",
            expected=exp,
            actual=None,
            message_ru=f"Слог '{exp}' пропущен",
        )
    # extra_syllable
    return FeedbackItem(
        index=seq_index,
        type="extra_syllable",
        expected=None,
        actual=act,
        message_ru=f"Лишний слог '{act}'",
    )


def feedback_for_alignment(steps: list[AlignmentStep]) -> list[FeedbackItem]:
    """
    Convert alignment trace to API feedback.

    We skip verbose rows for fully correct tokens if you want a compact UI —
    here we include 'correct' entries so the client can highlight syllables; filter client-side if needed.
    """
    out: list[FeedbackItem] = []
    for i, step in enumerate(steps):
        out.append(_msg_for_step(step, i))
    return out
