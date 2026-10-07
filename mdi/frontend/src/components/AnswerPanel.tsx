import React from 'react';
import { Bot, AlertCircle, FileCheck, Tag } from 'lucide-react';
import { QueryResponse } from '../services/api';

interface AnswerPanelProps {
  response: QueryResponse | null;
  loading: boolean;
  onSelectEvidence: (elementId: string) => void;
}

export const AnswerPanel: React.FC<AnswerPanelProps> = ({
  response,
  loading,
  onSelectEvidence,
}) => {
  if (loading) {
    return (
      <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-6 shadow-lg backdrop-blur flex flex-col items-center justify-center min-h-[160px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-400 mb-3"></div>
        <p className="text-xs text-slate-400 font-medium">Running Vector RAG + BM25 Hybrid Retrieval & Reranking...</p>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-6 shadow-lg backdrop-blur text-center min-h-[160px] flex flex-col items-center justify-center">
        <Bot size={32} className="text-slate-600 mb-2" />
        <p className="text-sm text-slate-400">Ask a question above to analyze your documents and retrieve verified evidence.</p>
      </div>
    );
  }

  const isNotFound = response.answer === 'Not found in the provided documents.';

  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-5 shadow-lg backdrop-blur space-y-4">
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <Bot size={18} className="text-indigo-400" />
          AI Reasoning & Answer
        </h3>
        {isNotFound ? (
          <span className="flex items-center gap-1 text-[11px] bg-rose-950 text-rose-400 px-2 py-0.5 rounded border border-rose-800">
            <AlertCircle size={12} /> Insufficient Evidence
          </span>
        ) : (
          <span className="flex items-center gap-1 text-[11px] bg-emerald-950 text-emerald-400 px-2 py-0.5 rounded border border-emerald-800">
            <FileCheck size={12} /> Grounded Answer
          </span>
        )}
      </div>

      <div className={`p-4 rounded-lg text-sm leading-relaxed ${
        isNotFound 
          ? 'bg-rose-950/40 border border-rose-900/60 text-rose-200 font-medium' 
          : 'bg-slate-900/80 border border-slate-700/50 text-slate-100'
      }`}>
        {response.answer}
      </div>

      {response.claims && response.claims.length > 0 && !isNotFound && (
        <div className="mt-3 space-y-2">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Tag size={12} className="text-indigo-400" />
            Verified Claims ({response.claims.length})
          </h4>
          <div className="space-y-1.5">
            {response.claims.map((claim, idx) => (
              <div key={idx} className="bg-slate-900/50 border border-slate-700/40 p-2.5 rounded text-xs text-slate-300">
                <p>{claim.text}</p>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {claim.evidence_ids.map((eid) => (
                    <button
                      key={eid}
                      onClick={() => onSelectEvidence(eid)}
                      className="text-[10px] bg-indigo-950 hover:bg-indigo-900 text-indigo-300 border border-indigo-700/60 px-2 py-0.5 rounded font-mono transition-all"
                    >
                      Ref: {eid}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
