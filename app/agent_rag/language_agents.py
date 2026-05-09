"""Языки чата: коллекции Qdrant, префиксы режима практики, промпты перевода."""

from __future__ import annotations

import os
from typing import Literal

LanguageCode = Literal["zh", "fr", "es", "en"]

LANGUAGE_CODES: tuple[LanguageCode, ...] = ("zh", "fr", "es", "en")


def normalize_language(lang: str) -> LanguageCode:
    key = lang.lower().strip()
    if key not in LANGUAGE_CODES:
        raise ValueError(f"Неподдерживаемый язык агента: {lang!r}; допустимо: {LANGUAGE_CODES}")
    return key  # type: ignore[return-value]


def qdrant_collection_for_language(lang: LanguageCode) -> str:
    """Имя коллекции Qdrant для языка (можно переопределить через env)."""
    if lang == "zh":
        return (
            os.getenv("QDRANT_COLLECTION_ZH") or os.getenv("QDRANT_COLLECTION") or "chinese_lexicon"
        ).strip()
    if lang == "fr":
        return (os.getenv("QDRANT_COLLECTION_FR") or "french_lexicon").strip()
    if lang == "es":
        return (os.getenv("QDRANT_COLLECTION_ES") or "spanish_lexicon").strip()
    return (os.getenv("QDRANT_COLLECTION_EN") or "english_lexicon").strip()


PRACTICE_MESSAGE_PREFIX: dict[LanguageCode, str] = {
    "zh": (
        "Ты собеседник для практики китайского. Отвечай на упрощённом китайском, "
        "естественно и коротко, 1-3 предложения. Не переводи на русский, если тебя явно не попросили.\n"
        "Сообщение пользователя: "
    ),
    "fr": (
        "Tu es un partenaire de conversation pour pratiquer le français. Réponds en français, "
        "naturellement et en 1–3 phrases courtes. Ne passe pas au russe sauf si on te le demande.\n"
        "Message de l'utilisateur : "
    ),
    "es": (
        "Eres un compañero para practicar español. Responde en español, de forma natural "
        "y en 1–3 frases cortas. No cambies al ruso salvo que te lo pidan.\n"
        "Mensaje del usuario: "
    ),
    "en": (
        "You are a conversation partner for English practice. Reply in English, naturally "
        "and in 1–3 short sentences. Do not switch to Russian unless asked.\n"
        "User message: "
    ),
}

TRANSLATE_TO_RUSSIAN_SYSTEM: dict[LanguageCode, str] = {
    "zh": "Переведи текст с китайского на русский. Верни только перевод, без пояснений и кавычек.",
    "fr": "Переведи текст с французского на русский. Верни только перевод, без пояснений и кавычек.",
    "es": "Переведи текст с испанского на русский. Верни только перевод, без пояснений и кавычек.",
    "en": "Переведи текст с английского на русский. Верни только перевод, без пояснений и кавычек.",
}
