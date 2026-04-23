"""
Сравнение ожидаемой фразы и распознанных иероглифов (отдельно от слогового скоринга).

Нормализация: без пробелов, без частой пунктуации, NFC.
"""

from __future__ import annotations

import re
import unicodedata


def _normalize_phrase(s: str) -> str:
    s = unicodedata.normalize("NFC", (s or "").strip())
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。！？、；：·…\u3000「」『』]", "", s)
    return s


def evaluate_text_match(expected: str, recognized: str) -> tuple[bool, str]:
    """
    Возвращает (совпадение после нормализации, короткое пояснение по-русски).

    Если эталон пуст — совпадение False.
    """
    ne = _normalize_phrase(expected)
    nr = _normalize_phrase(recognized)

    if not ne:
        return False, "Задайте ожидаемую фразу (汉字)."

    if not nr:
        return False, "Модель не вернула текст — попробуйте записать громче или ближе к микрофону."

    if ne == nr:
        return (
            True,
            "Распознанный текст совпадает с ожидаемым — по транскрипции произношение верное.",
        )

    return (
        False,
        f"Текст не совпал: ожидалось «{expected.strip()}», модель услышала «{recognized.strip()}».",
    )
