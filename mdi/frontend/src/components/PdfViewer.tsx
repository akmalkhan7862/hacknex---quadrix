import React, { useState } from 'react';
import { FileText, ChevronLeft, ChevronRight, Eye } from 'lucide-react';
import { DocumentItem } from '../services/api';

interface PdfViewerProps {
  document: DocumentItem | null;
  targetPage?: number;
}

export const PdfViewer: React.FC<PdfViewerProps> = ({ document, targetPage = 1 }) => {
  const [currentPage, setCurrentPage] = useState<number>(targetPage);

  if (!document) {
    return (
      <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-6 shadow-lg backdrop-blur flex flex-col items-center justify-center min-h-[300px] text-center">
        <FileText size={40} className="text-slate-600 mb-2" />
        <p className="text-sm text-slate-400 font-medium">Select a document from the list to preview</p>
        <p className="text-xs text-slate-500 mt-1">PDF Viewer supports page jump & provenance inspection</p>
      </div>
    );
  }

  // File URL served by FastAPI static mount
  const fileUrl = `http://localhost:8000/files/${document.doc_id}_${document.document_title}#page=${currentPage}`;

  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-4 shadow-lg backdrop-blur flex flex-col h-[500px]">
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-3 mb-3">
        <div className="flex items-center gap-2 overflow-hidden">
          <Eye size={18} className="text-indigo-400 shrink-0" />
          <span className="text-sm font-semibold text-slate-200 truncate">
            {document.document_title}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage <= 1}
            className="p-1 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-40 text-slate-200 transition-all"
          >
            <ChevronLeft size={16} />
          </button>
          <span className="text-xs font-mono text-slate-300">
            Page {currentPage} / {document.page_count || 1}
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.min(document.page_count || 1, p + 1))}
            disabled={currentPage >= (document.page_count || 1)}
            className="p-1 rounded bg-slate-700 hover:bg-slate-600 disabled:opacity-40 text-slate-200 transition-all"
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      <div className="flex-1 bg-slate-950 rounded-lg overflow-hidden border border-slate-700/50">
        <iframe
          src={fileUrl}
          className="w-full h-full border-0"
          title={document.document_title}
        />
      </div>
    </div>
  );
};
