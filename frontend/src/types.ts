export interface Source {
  source: string;
  title: string | null;
  score: number;
  content: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
}
