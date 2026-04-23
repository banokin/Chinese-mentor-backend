"""
Chinese text → syllables with tone numbers (TONE3), using pypinyin.

Junior note: we use one syllable per 汉字 by default — good enough for learner drills;
phrase-level segmentation can be added later.
"""

from __future__ import annotations

import re

from pypinyin import Style, lazy_pinyin


_TONE_DIGIT = re.compile(r"([1-5])$")


def strip_tone(syllable: str) -> str:
    """Remove trailing tone digit if present (e.g. ni3 -> ni)."""
    return _TONE_DIGIT.sub("", syllable.lower())


def text_to_tone3_syllables(text: str) -> list[str]:
    """
    Convert Chinese (and ignore non-Han for alignment gaps) to list like ['ni3', 'hao3'].

    Non-Chinese characters are dropped for syllable lists — scoring compares Han coverage.
    """
    normalized = text.strip()
    if not normalized:
        return []

    syllables: list[str] = []
    for ch in normalized:
        # Skip spaces and punctuation for syllable stream
        if not ("\u4e00" <= ch <= "\u9fff"):
            continue
        # lazy_pinyin returns list of one str per char
        py = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)
        if py:
            syllables.append(str(py[0]).lower())
    return syllables
