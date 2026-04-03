import type { Source } from "./types";

export interface SSEEvent {
  type: "token" | "sources" | "done";
  content?: string;
  sources?: Source[];
}

export async function* streamChat(
  message: string,
  signal?: AbortSignal
): AsyncGenerator<SSEEvent> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const event: SSEEvent = JSON.parse(line.slice(6));
        yield event;
      }
    }
  }
}
