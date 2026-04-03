import { useState } from "react";
import type { Source } from "../types";

interface SourceCardProps {
  sources: Source[];
}

export function SourceCard({ sources }: SourceCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (sources.length === 0) return null;

  return (
    <div className="source-card">
      <button
        className="source-toggle"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="source-toggle-icon">{isExpanded ? "\u25BC" : "\u25B6"}</span>
        Sources ({sources.length})
      </button>
      {isExpanded && (
        <div className="source-list">
          {sources.map((s, i) => (
            <div key={i} className="source-item">
              <div className="source-header">
                <span className="source-title">{s.title ?? s.source}</span>
                <span className="source-score">
                  {(s.score * 100).toFixed(1)}%
                </span>
              </div>
              <p className="source-content">{s.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
