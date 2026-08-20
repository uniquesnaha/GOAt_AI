import { useRef, useState } from "react";
import { openSttSocket, type Language } from "../lib/api";
import { MicPcmStreamer } from "../lib/audio";

interface MicButtonProps {
  language: Language;
  disabled?: boolean;
  onPartial: (text: string) => void;
  onFinal: (text: string, sttLatencyMs?: number) => void;
  onError: (message: string) => void;
}

export function MicButton({ language, disabled, onPartial, onFinal, onError }: MicButtonProps) {
  const [recording, setRecording] = useState(false);
  const streamerRef = useRef<MicPcmStreamer | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const start = async () => {
    if (recording || disabled) return;

    try {
      const ws = openSttSocket(language, (evt) => {
        if (evt.event === "error") {
          onError(evt.text ?? "STT error");
          stop();
          return;
        }
        if (evt.event === "transcript.partial" && evt.text) {
          onPartial(evt.text);
        }
        if (evt.event === "transcript.final" && evt.text) {
          onFinal(evt.text, evt.stt_latency_ms);
        }
      });
      wsRef.current = ws;

      const streamer = new MicPcmStreamer();
      await streamer.start((chunk) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(chunk);
        }
      });
      streamerRef.current = streamer;

      setRecording(true);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Microphone access failed");
      stop();
    }
  };

  const stop = () => {
    streamerRef.current?.stop();
    streamerRef.current = null;

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send("end");
      wsRef.current.close();
    }
    wsRef.current = null;

    setRecording(false);
  };

  return (
    <button
      type="button"
      disabled={disabled}
      onMouseDown={start}
      onMouseUp={stop}
      onMouseLeave={() => recording && stop()}
      onTouchStart={(e) => {
        e.preventDefault();
        start();
      }}
      onTouchEnd={(e) => {
        e.preventDefault();
        stop();
      }}
      className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-full border transition-all ${
        recording
          ? "border-red-400/50 bg-red-500/20 text-red-300 scale-110"
          : "border-goat-border bg-goat-panel text-slate-300 hover:border-goat-accent/50 hover:text-goat-accent"
      } disabled:cursor-not-allowed disabled:opacity-40`}
      title="Hold to talk"
    >
      {recording ? "●" : "🎙"}
    </button>
  );
}
