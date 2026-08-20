"""
WS /api/stt/stream?language=ta|hi

Bridges the browser's mic audio (raw PCM16 mono frames, 16kHz, sent as
binary WebSocket messages) to Sarvam's realtime streaming STT session, and
relays partial/final transcript events back to the browser as JSON.

This is STT plumbing only — it never touches the RAG engine. STT latency
(first audio chunk sent -> final transcript received) is measured here and
reported as its own field, kept separate from RAG latency per the plan's
"don't relabel STT latency as RAG latency" rule.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.stt.sarvam_client import SarvamStreamingSTT

router = APIRouter()


@router.websocket("/api/stt/stream")
async def stt_stream(websocket: WebSocket) -> None:
    language = websocket.query_params.get("language", "ta")

    await websocket.accept()

    try:
        async with SarvamStreamingSTT(language=language) as stt:
            first_audio_at: float | None = None

            async def forward_transcripts() -> None:
                async for event in stt.events():
                    payload = {
                        "event": event.event,
                        "text": event.text,
                        "is_final": event.is_final,
                    }
                    if event.is_final and first_audio_at is not None:
                        payload["stt_latency_ms"] = (time.perf_counter() - first_audio_at) * 1000
                    await websocket.send_json(payload)

            forward_task = asyncio.create_task(forward_transcripts())

            try:
                while True:
                    message = await websocket.receive()

                    if message.get("type") == "websocket.disconnect":
                        break

                    audio_bytes = message.get("bytes")
                    if audio_bytes is not None:
                        if first_audio_at is None:
                            first_audio_at = time.perf_counter()
                        await stt.send_audio_chunk(audio_bytes)
                        continue

                    text = message.get("text")
                    if text == "end":
                        break

            finally:
                forward_task.cancel()
                try:
                    await forward_task
                except (asyncio.CancelledError, Exception):
                    pass

    except WebSocketDisconnect:
        return
    except RuntimeError as exc:
        # e.g. SARVAM_API_KEY not configured
        await websocket.send_json({"event": "error", "text": str(exc)})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
