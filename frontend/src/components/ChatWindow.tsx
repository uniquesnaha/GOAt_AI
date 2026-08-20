import { useEffect, useRef } from "react";
import { MessageBubble, type ChatMessage } from "./MessageBubble";

export function ChatWindow({ messages }: { messages: ChatMessage[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, messages[messages.length - 1]?.text]);

  if (messages.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-slate-500">
        <div className="text-5xl">🐐</div>
        <p className="text-lg font-medium text-slate-300">GOAt AI</p>
        <p className="max-w-sm text-sm">
          Ask a question in Tamil or Hindi — type or press the mic. Answers are
          grounded strictly in the retrieved knowledge base.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 overflow-y-auto px-1 py-4">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
