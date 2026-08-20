import { useState } from "react";
import { ChatWindow } from "./components/ChatWindow";
import type { ChatMessage } from "./components/MessageBubble";
import { MicButton } from "./components/MicButton";
import { postQuery, type Language } from "./lib/api";

const LANGUAGES: { code: Language; label: string }[] = [
  { code: "ta", label: "தமிழ்" },
  { code: "hi", label: "हिन्दी" },
];

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

export default function App() {
  const [language, setLanguage] = useState<Language>("ta");
  const [input, setInput] = useState("");
  const [partialTranscript, setPartialTranscript] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sending, setSending] = useState(false);
  const [sttError, setSttError] = useState<string | null>(null);

  const send = async (text: string) => {
    const query = text.trim();
    if (!query || sending) return;

    setInput("");
    setPartialTranscript("");

    const userMessage: ChatMessage = { id: uid(), role: "user", text: query };
    const pendingId = uid();
    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: pendingId, role: "assistant", text: "", pending: true },
    ]);
    setSending(true);

    try {
      const response = await postQuery(query, language);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingId
            ? {
                id: pendingId,
                role: "assistant",
                text: response.answer,
                sources: response.sources,
                stageLatencies: response.stage_latencies,
                guardrail: response.guardrail,
                retried: response.retried,
              }
            : m
        )
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingId
            ? {
                id: pendingId,
                role: "assistant",
                text: err instanceof Error ? err.message : "Request failed.",
                guardrail: { blocked: true, code: "request_error" },
              }
            : m
        )
      );
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="mx-auto flex h-screen max-w-3xl flex-col px-4">
      <header className="flex items-center justify-between border-b border-goat-border py-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🐐</span>
          <div>
            <h1 className="text-base font-semibold tracking-tight">GOAt AI</h1>
            <p className="text-xs text-slate-500">Grounded RAG · &lt;200ms target</p>
          </div>
        </div>

        <div className="flex gap-1 rounded-full border border-goat-border bg-goat-panel p-1">
          {LANGUAGES.map((lang) => (
            <button
              key={lang.code}
              onClick={() => setLanguage(lang.code)}
              className={`rounded-full px-3 py-1 text-sm transition-colors ${
                language === lang.code
                  ? "bg-goat-accent2/30 text-goat-accent2"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {lang.label}
            </button>
          ))}
        </div>
      </header>

      <main className="flex-1 overflow-hidden">
        <ChatWindow messages={messages} />
      </main>

      {sttError && (
        <div className="mb-2 rounded-lg border border-red-400/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
          {sttError}
        </div>
      )}

      <footer className="flex items-center gap-2 border-t border-goat-border py-4">
        <MicButton
          language={language}
          disabled={sending}
          onPartial={(text) => setPartialTranscript(text)}
          onFinal={(text) => {
            setPartialTranscript("");
            void send(text);
          }}
          onError={(msg) => setSttError(msg)}
        />

        <input
          value={partialTranscript || input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void send(input);
          }}
          placeholder={language === "ta" ? "உங்கள் கேள்வியை உள்ளிடவும்..." : "अपना प्रश्न लिखें..."}
          disabled={sending}
          className="flex-1 rounded-full border border-goat-border bg-goat-panel px-4 py-2.5 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-goat-accent/50 disabled:opacity-50"
        />

        <button
          onClick={() => void send(input)}
          disabled={sending || !input.trim()}
          className="rounded-full bg-goat-accent2/90 px-5 py-2.5 text-sm font-medium text-slate-900 transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-30"
        >
          Send
        </button>
      </footer>
    </div>
  );
}
