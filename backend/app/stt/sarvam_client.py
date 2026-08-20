"""
Sarvam realtime streaming STT client.

Built against Sarvam's documented WebSocket contract
(wss://api.sarvam.ai/speech-to-text-realtime/ws, model=saaras:v3-realtime):

  - auth via the `API-SUBSCRIPTION-KEY` header
  - query params: language_code (BCP-47, e.g. "ta-IN"/"hi-IN"), model,
    sample_rate, encoding, stream_type, mode, endpointing
  - client sends `{"event": "audio_input", "audio": "<base64 PCM16>"}` for
    each audio chunk and `{"event": "end"}` to close the turn
  - server sends `session.begin`, `transcript.partial`, `transcript.final`,
    `session.end`, and `error` events

This module only relays bytes/events between our own WebSocket route and
Sarvam's — it does not do any RAG logic, so it's not covered by the
"don't change RAG logic" constraint at all.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import websockets

from app.config import settings


SARVAM_WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"

LANGUAGE_TO_BCP47 = {
    "ta": "ta-IN",
    "hi": "hi-IN",
}


@dataclass
class TranscriptEvent:
    event: str  # "session.begin" | "transcript.partial" | "transcript.final" | "session.end" | "error"
    text: Optional[str] = None
    is_final: bool = False
    raw: Optional[dict] = None


class SarvamStreamingSTT:
    """One streaming session, one browser connection <-> one Sarvam session."""

    def __init__(self, language: str, sample_rate: int = 16000):
        if language not in LANGUAGE_TO_BCP47:
            raise ValueError(f"Unsupported STT language: {language}")

        if not settings.sarvam_api_key:
            raise RuntimeError(
                "SARVAM_API_KEY is not set — streaming STT is unavailable."
            )

        self._language = language
        self._sample_rate = sample_rate
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    async def __aenter__(self) -> "SarvamStreamingSTT":
        params = {
            "language_code": LANGUAGE_TO_BCP47[self._language],
            "model": "saaras:v3-realtime",
            "sample_rate": str(self._sample_rate),
            "encoding": "linear16",
            "stream_type": "balanced",
            "mode": "transcribe",
            "endpointing": "vad",
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())

        self._ws = await websockets.connect(
            f"{SARVAM_WS_URL}?{query}",
            extra_headers={"API-SUBSCRIPTION-KEY": settings.sarvam_api_key},
            ping_interval=20,
            ping_timeout=20,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"event": "end"}))
            except Exception:
                pass
            await self._ws.close()

    async def send_audio_chunk(self, pcm16_bytes: bytes) -> None:
        payload = {
            "event": "audio_input",
            "audio": base64.b64encode(pcm16_bytes).decode("ascii"),
        }
        await self._ws.send(json.dumps(payload))

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        async for raw_message in self._ws:
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                continue

            event = data.get("event")

            if event == "transcript.partial":
                yield TranscriptEvent(event=event, text=data.get("text"), is_final=False, raw=data)
            elif event == "transcript.final":
                yield TranscriptEvent(event=event, text=data.get("text"), is_final=True, raw=data)
            elif event in ("session.begin", "session.end", "error"):
                yield TranscriptEvent(event=event, raw=data)
