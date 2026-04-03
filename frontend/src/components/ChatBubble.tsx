import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "../types";
import { SourceCard } from "./SourceCard";
import { TypingIndicator } from "./TypingIndicator";

interface ChatBubbleProps {
  message: Message;
}

export function ChatBubble({ message }: ChatBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
      <div className="bubble-label">{isUser ? "You" : "Assistant"}</div>
      <div className="bubble-content">
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <>
            {message.content ? (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            ) : message.isStreaming ? (
              <TypingIndicator />
            ) : null}
          </>
        )}
      </div>
      {!isUser && message.sources && message.sources.length > 0 && (
        <SourceCard sources={message.sources} />
      )}
    </div>
  );
}
