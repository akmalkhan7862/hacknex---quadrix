import React, { useState } from 'react';
import { Send, Sparkles, Loader2 } from 'lucide-react';

interface ChatPanelProps {
  onAskQuestion: (question: string) => void;
  loading: boolean;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({ onAskQuestion, loading }) => {
  const [question, setQuestion] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.strip ? question.trim() : question) return;
    onAskQuestion(question.trim());
  };

  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-4 shadow-lg backdrop-blur">
      <form onSubmit={handleSubmit} className="flex items-center gap-3">
        <div className="relative flex-1">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about the uploaded documents..."
            disabled={loading}
            className="w-full bg-slate-900 border border-slate-700 focus:border-indigo-500 rounded-lg py-2.5 pl-10 pr-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all"
          />
          <Sparkles size={16} className="absolute left-3 top-3 text-indigo-400" />
        </div>
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-medium px-5 py-2.5 rounded-lg text-sm flex items-center gap-2 transition-all shadow-md shrink-0"
        >
          {loading ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Send size={16} />
          )}
          <span>{loading ? 'Analyzing...' : 'Ask'}</span>
        </button>
      </form>
    </div>
  );
};
