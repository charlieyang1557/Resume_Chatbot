const STARTERS = [
  {
    text: "How did Charlie reduce onboarding time by 30%?",
    icon: "\u{1F4CA}",
  },
  {
    text: "Tell me about the anomaly detection work at PTC",
    icon: "\u{1F50D}",
  },
  {
    text: "What was Charlie's impact at Pinecone?",
    icon: "\u{1F3AF}",
  },
];

interface StarterQuestionsProps {
  onSelect: (question: string) => void;
}

export function StarterQuestions({ onSelect }: StarterQuestionsProps) {
  return (
    <div className="starter-questions">
      <p className="starter-label">Try asking</p>
      <div className="starter-cards">
        {STARTERS.map((s) => (
          <button
            key={s.text}
            className="starter-card"
            onClick={() => onSelect(s.text)}
          >
            <span className="starter-icon">{s.icon}</span>
            <span className="starter-text">{s.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
