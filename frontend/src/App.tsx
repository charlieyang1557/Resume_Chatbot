import { ChatWindow } from "./components/ChatWindow";
import { Hero } from "./components/Hero";
import { InputBar } from "./components/InputBar";
import { StarterQuestions } from "./components/StarterQuestions";
import { ThemeToggle } from "./components/ThemeToggle";
import { useChat } from "./hooks/useChat";
import { useTheme } from "./hooks/useTheme";

export default function App() {
  const { messages, isLoading, sendMessage } = useChat();
  const { theme, toggle } = useTheme();
  const isEmpty = messages.length === 0;

  return (
    <div className="app">
      <ThemeToggle theme={theme} onToggle={toggle} />
      <div className="container">
        <Hero />
        <div className="chat-container">
          {isEmpty ? (
            <StarterQuestions onSelect={sendMessage} />
          ) : (
            <ChatWindow messages={messages} />
          )}
          <InputBar onSend={sendMessage} disabled={isLoading} />
        </div>
      </div>
    </div>
  );
}
