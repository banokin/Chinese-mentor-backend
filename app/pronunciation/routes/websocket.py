"""WebSocket /ws/pronunciation — MVP buffered realtime (no WebRTC / DSP)."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.asr_service import ASRError, transcribe_bytes
from app.pronunciation.services.websocket_session_service import (
    PronunciationSessionState,
    append_chunk,
    build_final_result,
    handle_start,
    run_partial_transcription,
    snapshot_buffer,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


@router.websocket("/ws/pronunciation")
async def pronunciation_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    state = PronunciationSessionState()

    await _send_json(websocket, {"type": "status", "message": "connected"})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_json(
                    websocket,
                    {"type": "error", "message": "Некорректный JSON"},
                )
                continue

            mtype = msg.get("type")
            if mtype == "start":
                expected = str(msg.get("expected_text", ""))
                await handle_start(state, expected)
                await _send_json(websocket, {"type": "status", "message": "session_started"})
            elif mtype == "audio_chunk":
                chunk_b64 = msg.get("chunk_base64", "")
                if not isinstance(chunk_b64, str):
                    await _send_json(websocket, {"type": "error", "message": "chunk_base64 required"})
                    continue
                try:
                    buf_snapshot = await append_chunk(state, chunk_b64)
                except ValueError as e:
                    await _send_json(websocket, {"type": "error", "message": str(e)})
                    continue
                if buf_snapshot:
                    try:
                        text, _meta = await run_partial_transcription(buf_snapshot)
                        if text:
                            await _send_json(
                                websocket,
                                {"type": "partial_result", "recognized_text": text},
                            )
                    except ASRError:
                        # Partial failures are non-fatal for MVP
                        pass
            elif mtype == "stop":
                expected_text, audio_bytes = await snapshot_buffer(state)
                if not audio_bytes:
                    await _send_json(
                        websocket,
                        {"type": "error", "message": "Пустой буфер аудио"},
                    )
                    continue
                try:
                    recognized, _meta = await transcribe_bytes(
                        audio_bytes,
                        filename="session.webm",
                    )
                except ASRError as e:
                    await _send_json(
                        websocket,
                        {"type": "error", "message": e.message},
                    )
                    continue
                data = await build_final_result(expected_text, recognized)
                await _send_json(websocket, {"type": "final_result", "data": data})
            else:
                await _send_json(
                    websocket,
                    {"type": "error", "message": f"Неизвестный type: {mtype!r}"},
                )
    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket handler error")
        try:
            await _send_json(
                websocket,
                {"type": "error", "message": "Внутренняя ошибка сервера"},
            )
        except Exception:
            pass
