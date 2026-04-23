"""HTTP routes for LangChain RAG agent chat."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent_rag.agent import run_agent_query


class AgentMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., min_length=1, description="Message text")


class AgentRequest(BaseModel):
    messages: list[AgentMessage] = Field(..., min_length=1)


class AgentResponse(BaseModel):
    message: str


router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/", response_model=AgentResponse)
async def agent_chat(payload: AgentRequest) -> AgentResponse:
    messages = payload.messages
    last_user_idx = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"), -1)
    if last_user_idx < 0:
        raise HTTPException(status_code=400, detail={"message": "Не найдено сообщение пользователя"})

    question = messages[last_user_idx].content.strip()
    if not question:
        raise HTTPException(status_code=400, detail={"message": "Пустой запрос"})

    history = [m.model_dump() for m in messages[:last_user_idx]]
    answer = await run_agent_query(question, chat_history=history, verbose=False)
    if not answer:
        raise HTTPException(status_code=502, detail={"message": "Агент вернул пустой ответ"})
    return AgentResponse(message=answer)
