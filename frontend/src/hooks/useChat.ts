import { useCallback, useRef, useState } from "react";
import { streamChat } from "../api";
import type { Message, Source } from "../types";

let idCounter = 0;
function nextId(): string {
  return `msg-${++idCounter}-${Date.now()}`;
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (text: string) => {
    const userMsg: Message = { id: nextId(), role: "user", content: text };
    const assistantId = nextId();
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let accumulated = "";
      let sources: Source[] = [];

      for await (const event of streamChat(text, controller.signal)) {
        if (event.type === "token" && event.content) {
          accumulated += event.content;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: accumulated } : m
            )
          );
        } else if (event.type === "sources" && event.sources) {
          sources = event.sources;
        } else if (event.type === "done") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: accumulated, sources, isStreaming: false }
                : m
            )
          );
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: "Sorry, something went wrong. Please try again.",
                  isStreaming: false,
                }
              : m
          )
        );
      }
    } finally {
      setIsLoading(false);
      abortRef.current = null;
    }
  }, []);

  return { messages, isLoading, sendMessage };
}
