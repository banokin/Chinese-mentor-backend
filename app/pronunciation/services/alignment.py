"""Syllable alignment + score aggregation (no feedback text — keeps imports acyclic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.pronunciation.schemas import FinalScores
from app.services.pinyin_service import strip_tone

Op = Literal["correct", "tone_error", "syllable_error", "missing_syllable", "extra_syllable"]


@dataclass(frozen=True)
class AlignmentStep:
    """One step in the optimal syllable alignment."""

    op: Op
    expected: str | None
    actual: str | None
    expected_index: int


def _substitution_cost(exp: str, act: str) -> tuple[int, Op]:
    if exp == act:
        return 0, "correct"
    if strip_tone(exp) == strip_tone(act):
        return 1, "tone_error"
    return 2, "syllable_error"


def align_syllables(expected: list[str], actual: list[str]) -> list[AlignmentStep]:
    """Align syllable lists with DP (see scoring_service docstring for cost model)."""
    n, m = len(expected), len(actual)
    inf = 10**9
    dp: list[list[int]] = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + 2
        back[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + 2
        back[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost, _ = _substitution_cost(expected[i - 1], actual[j - 1])
            diag = dp[i - 1][j - 1] + sub_cost
            up = dp[i - 1][j] + 2
            left = dp[i][j - 1] + 2
            best = min(diag, up, left)
            dp[i][j] = best
            if best == diag:
                back[i][j] = "diag"
            elif best == up:
                back[i][j] = "up"
            else:
                back[i][j] = "left"

    steps: list[AlignmentStep] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and back[i][j] == "diag":
            _, op = _substitution_cost(expected[i - 1], actual[j - 1])
            steps.append(
                AlignmentStep(
                    op=op,
                    expected=expected[i - 1],
                    actual=actual[j - 1],
                    expected_index=i - 1,
                )
            )
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or back[i][j] == "up"):
            steps.append(
                AlignmentStep(
                    op="missing_syllable",
                    expected=expected[i - 1],
                    actual=None,
                    expected_index=i - 1,
                )
            )
            i -= 1
        elif j > 0:
            steps.append(
                AlignmentStep(
                    op="extra_syllable",
                    expected=None,
                    actual=actual[j - 1],
                    expected_index=-1,
                )
            )
            j -= 1
        else:
            break

    steps.reverse()
    return steps


def compute_scores(expected: list[str], actual: list[str], steps: list[AlignmentStep]) -> FinalScores:
    """Aggregate metrics from alignment."""
    exp_len = max(len(expected), 1)

    correct = sum(1 for s in steps if s.op == "correct")
    tone_err = sum(1 for s in steps if s.op == "tone_error")
    missing = sum(1 for s in steps if s.op == "missing_syllable")

    accuracy = correct / exp_len
    tone_denom = correct + tone_err
    tone_accuracy = (correct / tone_denom) if tone_denom else 1.0
    matched_or_tone = correct + tone_err
    completeness = max(0.0, min(1.0, matched_or_tone / exp_len))

    act_len = max(len(actual), 1)
    fluency = 1.0 - min(1.0, abs(len(actual) - len(expected)) / max(exp_len, act_len))
    fluency *= max(0.0, 1.0 - 0.15 * missing)
    fluency = max(0.0, min(1.0, fluency))

    return FinalScores(
        accuracy=max(0.0, min(1.0, accuracy)),
        tone_accuracy=max(0.0, min(1.0, tone_accuracy)),
        completeness=completeness,
        fluency=fluency,
    )
