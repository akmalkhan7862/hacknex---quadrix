import React from 'react';
import { Database, FileText, Table, ExternalLink } from 'lucide-react';
import { EvidenceSource } from '../services/api';

interface EvidencePanelProps {
  sources: EvidenceSource[];
  selectedEvidenceId: string | null;
  onSelectEvidence: (elementId: string, docName?: string, page?: number) => void;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  sources,
  selectedEvidenceId,
  onSelectEvidence,
}) => {
  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-5 shadow-lg backdrop-blur space-y-3">
      <h3 className="text-sm font-semibold text-slate-200 flex items-center justify-between border-b border-slate-700/60 pb-3">
        <span className="flex items-center gap-2">
          <Database size={18} className="text-indigo-400" />
          Retrieved Evidence & Provenance Sources
        </span>
        <span className="text-xs bg-indigo-950 text-indigo-300 border border-indigo-700/60 px-2 py-0.5 rounded-full font-medium">
          {sources.length} {sources.length === 1 ? 'Source' : 'Sources'}
        </span>
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-80 overflow-y-auto pr-1">
        {sources.map((src) => {
          const isSelected = selectedEvidenceId === src.element_id;
          return (
            <div
              key={src.element_id}
              onClick={() => onSelectEvidence(src.element_id, src.document, src.page)}
              className={`p-3 rounded-lg border text-xs cursor-pointer transition-all ${
                isSelected
                  ? 'bg-indigo-950/80 border-indigo-500 shadow-md ring-1 ring-indigo-500'
                  : 'bg-slate-900/60 border-slate-700/50 hover:bg-slate-900 hover:border-slate-600'
              }`}
            >
              <div className="flex items-center justify-between font-medium text-slate-200 mb-1.5">
                <div className="flex items-center gap-1.5 overflow-hidden">
                  {src.content_type === 'table' ? (
                    <Table size={14} className="text-amber-400 shrink-0" />
                  ) : (
                    <FileText size={14} className="text-indigo-400 shrink-0" />
                  )}
                  <span className="truncate">{src.document}</span>
                </div>
                <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-300 shrink-0 font-mono">
                  Page {src.page}
                </span>
              </div>

              <div className="text-[11px] text-indigo-300 font-medium mb-1 truncate">
                Section: {src.section}
              </div>

              {src.snippet && (
                <div className="bg-slate-950/60 p-2 rounded text-[11px] font-mono text-slate-400 border border-slate-800 line-clamp-3 overflow-hidden">
                  {src.snippet}
                </div>
              )}

              <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500 font-mono pt-1 border-t border-slate-800">
                <span>{src.element_id}</span>
                <span className="text-indigo-400 flex items-center gap-1 hover:underline">
                  View <ExternalLink size={10} />
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
