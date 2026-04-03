const STARTERS = [
  "How did Charlie reduce onboarding time by 30%?",
  "Tell me about the anomaly detection work at PTC",
  "What was Charlie's impact at Pinecone?",
];

interface StarterQuestionsProps {
  onSelect: (question: string) => void;
}

export function StarterQuestions({ onSelect }: StarterQuestionsProps) {
  return (
    <div className="starter-questions">
      <p className="starter-label">Try asking:</p>
      <div className="starter-cards">
        {STARTERS.map((q) => (
          <button key={q} className="starter-card" onClick={() => onSelect(q)}>
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
