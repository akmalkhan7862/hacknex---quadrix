import React, { useState, useEffect } from 'react';
import { Layers, ShieldCheck } from 'lucide-react';
import { api, DocumentItem, QueryResponse } from './services/api';
import { UploadPanel } from './components/UploadPanel';
import { DocumentList } from './components/DocumentList';
import { ChatPanel } from './components/ChatPanel';
import { AnswerPanel } from './components/AnswerPanel';
import { EvidencePanel } from './components/EvidencePanel';
import { PdfViewer } from './components/PdfViewer';

export const App: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);
  const [queryResponse, setQueryResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);

  const fetchDocuments = async () => {
    try {
      const data = await api.listDocuments();
      setDocuments(data.documents || []);
      if (!selectedDoc && data.documents && data.documents.length > 0) {
        setSelectedDoc(data.documents[0]);
      }
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleAskQuestion = async (question: string) => {
    setLoading(true);
    setSelectedEvidenceId(null);
    try {
      const res = await api.query(question);
      setQueryResponse(res);
    } catch (err: any) {
      console.error('Query error:', err);
      setQueryResponse({
        answer: 'Not found in the provided documents.',
        claims: [],
        sources: [],
        calculations: [],
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSelectEvidence = (elementId: string, docName?: string, page?: number) => {
    setSelectedEvidenceId(elementId);
    if (docName) {
      const matched = documents.find((d) => d.document_title === docName);
      if (matched) {
        setSelectedDoc(matched);
      }
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-900 text-slate-100 font-sans">
      {/* Top Navigation Bar */}
      <header className="bg-slate-800/90 border-b border-slate-700/80 px-6 py-3.5 flex items-center justify-between shadow-md backdrop-blur sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-600 rounded-lg shadow-inner">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold text-slate-100 tracking-tight">
              Multimodal Document Intelligence
            </h1>
            <p className="text-[11px] text-slate-400">
              HNX26PSI01 Specification • Hybrid Vector RAG + BM25 Engine
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-300 bg-slate-900/80 border border-slate-700 px-3 py-1.5 rounded-full">
          <ShieldCheck size={14} className="text-emerald-400" />
          <span>Strict Anti-Hallucination Evidence Gate Active</span>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload & Document List */}
        <div className="lg:col-span-4 flex flex-col gap-6 h-[calc(100vh-120px)]">
          <UploadPanel onUploadSuccess={fetchDocuments} />
          <DocumentList
            documents={documents}
            selectedDoc={selectedDoc}
            onSelectDoc={setSelectedDoc}
          />
        </div>

        {/* Right Column: Chat, Answer, Evidence & PDF Viewer */}
        <div className="lg:col-span-8 flex flex-col gap-6 overflow-y-auto pr-1 h-[calc(100vh-120px)]">
          <ChatPanel onAskQuestion={handleAskQuestion} loading={loading} />

          <AnswerPanel
            response={queryResponse}
            loading={loading}
            onSelectEvidence={handleSelectEvidence}
          />

          {queryResponse && queryResponse.sources && queryResponse.sources.length > 0 && (
            <EvidencePanel
              sources={queryResponse.sources}
              selectedEvidenceId={selectedEvidenceId}
              onSelectEvidence={handleSelectEvidence}
            />
          )}

          <PdfViewer document={selectedDoc} />
        </div>
      </div>
    </div>
  );
};

export default App;
