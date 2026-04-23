"""
High-level scoring: Chinese strings → syllables → alignment → feedback + scores.
"""

from __future__ import annotations

from app.pronunciation.schemas import FeedbackItem, FinalScores
from app.pronunciation.services.alignment import align_syllables, compute_scores
from app.pronunciation.services.feedback_service import feedback_for_alignment
from app.services.pinyin_service import text_to_tone3_syllables


def evaluate_expected_vs_recognized(expected_han: str, recognized_han: str) -> tuple[list[FeedbackItem], FinalScores]:
    """
    Full pipeline: Chinese → syllables → align → Russian feedback + aggregate scores.
    """
    exp = text_to_tone3_syllables(expected_han)
    act = text_to_tone3_syllables(recognized_han)
    steps = align_syllables(exp, act)
    scores = compute_scores(exp, act, steps)
    feedback = feedback_for_alignment(steps)
    return feedback, scores
