"""HTTP routes for LangChain RAG agent chat."""

from __future__ import annotations

import os
from typing import Literal

from openai import AsyncOpenAI

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.agent_rag.agent import run_agent_query
from app.agent_rag.ingest import ingest_documents_to_qdrant, load_uploaded_documents
from app.agent_rag.language_agents import (
    LanguageCode,
    PRACTICE_MESSAGE_PREFIX,
    TRANSLATE_TO_RUSSIAN_SYSTEM,
    qdrant_collection_for_language,
)


class AgentMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., min_length=1, description="Message text")


class AgentRequest(BaseModel):
    messages: list[AgentMessage] = Field(..., min_length=1)


class AgentResponse(BaseModel):
    message: str


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    source: LanguageCode = Field(default="zh", description="Язык исходного текста")


class TranslateResponse(BaseModel):
    translation: str


class RagUploadResponse(BaseModel):
    filename: str
    collection: str
    documents: int
    chunks: int
    vector_size: int
    points_count: int


class LanguageInfo(BaseModel):
    """Описание языка для клиентов (фронт, интеграции)."""

    code: LanguageCode
    label_ru: str
    practice_chat_path: str
    tutor_path: str
    default_qdrant_collection: str


router = APIRouter(prefix="/api/agent", tags=["agent"])
chat_router = APIRouter(prefix="/api/chat", tags=["chat"])
tutor_router = APIRouter(prefix="/api/tutor", tags=["tutor"])
languages_router = APIRouter(prefix="/api/languages", tags=["languages"])


def _language_catalog() -> list[LanguageInfo]:
    return [
        LanguageInfo(
            code="zh",
            label_ru="Китайский",
            practice_chat_path="/api/chat/zh",
            tutor_path="/api/tutor/zh",
            default_qdrant_collection=qdrant_collection_for_language("zh"),
        ),
        LanguageInfo(
            code="fr",
            label_ru="Французский",
            practice_chat_path="/api/chat/fr",
            tutor_path="/api/tutor/fr",
            default_qdrant_collection=qdrant_collection_for_language("fr"),
        ),
        LanguageInfo(
            code="es",
            label_ru="Испанский",
            practice_chat_path="/api/chat/es",
            tutor_path="/api/tutor/es",
            default_qdrant_collection=qdrant_collection_for_language("es"),
        ),
        LanguageInfo(
            code="en",
            label_ru="Английский",
            practice_chat_path="/api/chat/en",
            tutor_path="/api/tutor/en",
            default_qdrant_collection=qdrant_collection_for_language("en"),
        ),
    ]


@languages_router.get("", response_model=list[LanguageInfo])
async def list_languages() -> list[LanguageInfo]:
    """Список поддерживаемых языков и соответствующих HTTP-путей."""
    return _language_catalog()


def _openai_client() -> AsyncOpenAI:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail={"message": "OPENAI_API_KEY не задан"})
    return AsyncOpenAI(api_key=api_key)


async def _handle_agent_chat(
    payload: AgentRequest,
    *,
    language: LanguageCode = "zh",
    practice_mode: bool = False,
) -> AgentResponse:
    messages = payload.messages
    last_user_idx = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"), -1)
    if last_user_idx < 0:
        raise HTTPException(status_code=400, detail={"message": "Не найдено сообщение пользователя"})

    question = messages[last_user_idx].content.strip()
    if not question:
        raise HTTPException(status_code=400, detail={"message": "Пустой запрос"})

    history = [m.model_dump() for m in messages[:last_user_idx]]
    if practice_mode:
        question = PRACTICE_MESSAGE_PREFIX[language] + question

    answer = await run_agent_query(question, chat_history=history, verbose=False, language=language)
    if not answer:
        raise HTTPException(status_code=502, detail={"message": "Агент вернул пустой ответ"})
    return AgentResponse(message=answer)


@router.post("/", response_model=AgentResponse)
async def agent_chat(payload: AgentRequest) -> AgentResponse:
    """Китайский RAG-агент (полный режим репетитора). Совместимость: используйте также POST /api/tutor/zh."""
    return await _handle_agent_chat(payload, language="zh", practice_mode=False)


@tutor_router.post("/{language}", response_model=AgentResponse)
async def tutor_by_language(language: LanguageCode, payload: AgentRequest) -> AgentResponse:
    """RAG-агент — режим репетитора без короткого префикса «чат-практики»."""
    return await _handle_agent_chat(payload, language=language, practice_mode=False)


@chat_router.post("/", response_model=AgentResponse)
async def chat_zh_legacy(payload: AgentRequest) -> AgentResponse:
    """Практика китайского (префикс собеседника). Совместимость: то же, что POST /api/chat/zh."""
    return await _handle_agent_chat(payload, language="zh", practice_mode=True)


@chat_router.post("/translate", response_model=TranslateResponse)
async def translate_to_russian(payload: TranslateRequest) -> TranslateResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail={"message": "Пустой текст"})

    client = _openai_client()
    system_prompt = TRANSLATE_TO_RUSSIAN_SYSTEM[payload.source]
    try:
        response = await client.chat.completions.create(
            model=(os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail={"message": f"Ошибка перевода: {e}"}) from e

    translation = (response.choices[0].message.content or "").strip()
    if not translation:
        raise HTTPException(status_code=502, detail={"message": "Пустой ответ перевода"})
    return TranslateResponse(translation=translation)


@chat_router.post("/{language}", response_model=AgentResponse)
async def chat_practice_by_language(language: LanguageCode, payload: AgentRequest) -> AgentResponse:
    """Чат-практика для указанного языка (zh, fr, es, en)."""
    return await _handle_agent_chat(payload, language=language, practice_mode=True)


@router.post("/rag/upload", response_model=RagUploadResponse)
async def upload_rag_file(
    file: UploadFile = File(...),
    language: LanguageCode | None = Query(
        default=None,
        description=(
            "Если задан — документ индексируется в коллекцию этого языка "
            "(например fr → QDRANT_COLLECTION_FR / french_lexicon). "
            "Если нет — используется QDRANT_COLLECTION (как раньше)."
        ),
    ),
) -> RagUploadResponse:
    filename = file.filename or "document"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail={"message": "Файл пустой"})
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"message": "Файл больше 15 MB"})

    if language is not None:
        collection = qdrant_collection_for_language(language)
    else:
        collection = (os.getenv("QDRANT_COLLECTION") or "chinese_lexicon").strip()

    qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
    if not qdrant_url:
        raise HTTPException(status_code=503, detail={"message": "QDRANT_URL не задан"})
    embedding_model = (os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()

    try:
        documents = load_uploaded_documents(filename, content)
        result = ingest_documents_to_qdrant(
            documents,
            collection_name=collection,
            qdrant_url=qdrant_url,
            embedding_model=embedding_model,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"message": str(e)}) from e
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail={"message": str(e)}) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail={"message": f"Ошибка загрузки в RAG: {e}"}) from e

    return RagUploadResponse(filename=filename, **result)
