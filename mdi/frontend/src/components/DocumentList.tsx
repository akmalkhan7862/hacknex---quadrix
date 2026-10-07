import React from 'react';
import { FileText, CheckCircle2, Clock, AlertTriangle } from 'lucide-react';
import { DocumentItem } from '../services/api';

interface DocumentListProps {
  documents: DocumentItem[];
  selectedDoc: DocumentItem | null;
  onSelectDoc: (doc: DocumentItem) => void;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  selectedDoc,
  onSelectDoc,
}) => {
  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-4 shadow-lg backdrop-blur flex-1 flex flex-col min-h-0">
      <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center justify-between">
        <span>Uploaded Documents</span>
        <span className="text-xs bg-slate-700 px-2 py-0.5 rounded-full text-slate-300">
          {documents.length}
        </span>
      </h3>

      {documents.length === 0 ? (
        <div className="text-center text-xs text-slate-500 py-8">
          No documents uploaded yet. Upload a PDF to start.
        </div>
      ) : (
        <div className="overflow-y-auto space-y-2 pr-1 flex-1">
          {documents.map((doc) => {
            const isSelected = selectedDoc?.doc_id === doc.doc_id;
            return (
              <div
                key={doc.doc_id}
                onClick={() => onSelectDoc(doc)}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  isSelected
                    ? 'bg-indigo-950/60 border-indigo-500 shadow-md'
                    : 'bg-slate-900/60 border-slate-700/50 hover:bg-slate-900 hover:border-slate-600'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2 overflow-hidden">
                    <FileText size={16} className="text-indigo-400 shrink-0" />
                    <span className="text-xs font-medium text-slate-200 truncate">
                      {doc.document_title}
                    </span>
                  </div>
                  {doc.status === 'PROCESSED' && (
                    <CheckCircle2 size={14} className="text-emerald-400 shrink-0" />
                  )}
                  {doc.status === 'PROCESSING' && (
                    <Clock size={14} className="text-amber-400 animate-pulse shrink-0" />
                  )}
                  {doc.status === 'FAILED' && (
                    <AlertTriangle size={14} className="text-rose-400 shrink-0" />
                  )}
                </div>

                <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400">
                  <span>{doc.page_count} {doc.page_count === 1 ? 'Page' : 'Pages'}</span>
                  <span className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-semibold ${
                    doc.status === 'PROCESSED' ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' :
                    doc.status === 'PROCESSING' ? 'bg-amber-950 text-amber-400 border border-amber-800' :
                    'bg-rose-950 text-rose-400 border border-rose-800'
                  }`}>
                    {doc.status}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
