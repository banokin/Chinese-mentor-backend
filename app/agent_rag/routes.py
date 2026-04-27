"""HTTP routes for LangChain RAG agent chat."""

from __future__ import annotations

import os
from typing import Literal

from openai import AsyncOpenAI

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.agent_rag.agent import run_agent_query
from app.agent_rag.ingest import ingest_documents_to_qdrant, load_uploaded_documents


class AgentMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., min_length=1, description="Message text")


class AgentRequest(BaseModel):
    messages: list[AgentMessage] = Field(..., min_length=1)


class AgentResponse(BaseModel):
    message: str


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class TranslateResponse(BaseModel):
    translation: str


class RagUploadResponse(BaseModel):
    filename: str
    collection: str
    documents: int
    chunks: int
    vector_size: int
    points_count: int


router = APIRouter(prefix="/api/agent", tags=["agent"])
chat_router = APIRouter(prefix="/api/chat", tags=["chat"])


def _openai_client() -> AsyncOpenAI:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail={"message": "OPENAI_API_KEY не задан"})
    return AsyncOpenAI(api_key=api_key)


async def _handle_agent_chat(payload: AgentRequest, *, chinese_practice: bool = False) -> AgentResponse:
    messages = payload.messages
    last_user_idx = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"), -1)
    if last_user_idx < 0:
        raise HTTPException(status_code=400, detail={"message": "Не найдено сообщение пользователя"})

    question = messages[last_user_idx].content.strip()
    if not question:
        raise HTTPException(status_code=400, detail={"message": "Пустой запрос"})

    history = [m.model_dump() for m in messages[:last_user_idx]]
    if chinese_practice:
        question = (
            "Ты собеседник для практики китайского. Отвечай на упрощённом китайском, "
            "естественно и коротко, 1-3 предложения. Не переводи на русский, если тебя явно не попросили. "
            f"Сообщение пользователя: {question}"
        )
    answer = await run_agent_query(question, chat_history=history, verbose=False)
    if not answer:
        raise HTTPException(status_code=502, detail={"message": "Агент вернул пустой ответ"})
    return AgentResponse(message=answer)


@router.post("/", response_model=AgentResponse)
async def agent_chat(payload: AgentRequest) -> AgentResponse:
    return await _handle_agent_chat(payload)


@chat_router.post("/", response_model=AgentResponse)
async def chat(payload: AgentRequest) -> AgentResponse:
    return await _handle_agent_chat(payload, chinese_practice=True)


@chat_router.post("/translate", response_model=TranslateResponse)
async def translate_to_russian(payload: TranslateRequest) -> TranslateResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail={"message": "Пустой текст"})

    client = _openai_client()
    try:
        response = await client.chat.completions.create(
            model=(os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Переведи текст с китайского на русский. "
                        "Верни только перевод, без пояснений и кавычек."
                    ),
                },
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


@router.post("/rag/upload", response_model=RagUploadResponse)
async def upload_rag_file(file: UploadFile = File(...)) -> RagUploadResponse:
    filename = file.filename or "document"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail={"message": "Файл пустой"})
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"message": "Файл больше 15 MB"})

    collection = (os.getenv("QDRANT_COLLECTION") or "chinese_lexicon").strip()
    qdrant_url = (os.getenv("QDRANT_URL") or "http://localhost:6333").strip()
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
