'use client';

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import {
  FileText, Upload, RefreshCw, Trash2, ArrowLeft, ArrowRight,
  ChevronLeft, ChevronRight, AlertCircle, Check, HelpCircle, CornerDownLeft
} from 'lucide-react';
import { api, DocumentItem, Citation, QueryResponse } from '../../lib/api';

export default function WorkspacePage() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [activeViewerDocId, setActiveViewerDocId] = useState<string>('doc-q2-eff');
  const [activePageNum, setActivePageNum] = useState<number>(1);
  const [zoomLevel, setZoomLevel] = useState<number>(100);

  const [questionInput, setQuestionInput] = useState('');
  const [loadingQuery, setLoadingQuery] = useState(false);
  const [queryStage, setQueryStage] = useState('Idle');
  const [activeTab, setActiveTab] = useState<'evidence' | 'calculation' | 'reasoning'>('evidence');
  const [currentResponse, setCurrentResponse] = useState<QueryResponse | null>(null);
  const [highlightedCitationId, setHighlightedCitationId] = useState<string | null>(null);

  // Fetch Documents
  const loadDocuments = async () => {
    const docs = await api.listDocuments();
    setDocuments(docs);
    if (docs.length > 0 && selectedDocIds.length === 0) {
      setSelectedDocIds([docs[0].doc_id, docs[1]?.doc_id || docs[0].doc_id]);
      setActiveViewerDocId(docs[0].doc_id);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  // Handle Upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    try {
      await api.uploadDocument(file);
      await loadDocuments();
    } catch (err) {
      console.error(err);
    }
  };

  // Try Sample Documents Click
  const handleLoadSampleDocs = async () => {
    await loadDocuments();
    setSelectedDocIds(['doc-q2-eff', 'doc-q4-eff']);
    setActiveViewerDocId('doc-q2-eff');
  };

  // Run Query
  const handleRunQuery = async (queryText: string) => {
    if (!queryText.trim()) return;
    setLoadingQuery(true);
    setQueryStage('Analyzing question intent...');
    
    setTimeout(() => setQueryStage('Executing hybrid retrieval across vector & BM25...'), 200);
    setTimeout(() => setQueryStage('Reranking candidates & checking anti-hallucination gate...'), 500);

    try {
      const res = await api.query(queryText, selectedDocIds);
      setCurrentResponse(res);
      if (res.citations.length > 0) {
        setActiveViewerDocId(res.citations[0].document_id);
        setActivePageNum(res.citations[0].page);
        setHighlightedCitationId(res.citations[0].element_id);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingQuery(false);
      setQueryStage('Idle');
    }
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') {
        setActivePageNum((p) => Math.min(22, p + 1));
      } else if (e.key === 'ArrowLeft') {
        setActivePageNum((p) => Math.max(1, p - 1));
      } else if (e.key === 'Escape') {
        setHighlightedCitationId(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const activeDoc = documents.find((d) => d.doc_id === activeViewerDocId) || documents[0];
  const maxPages = activeDoc?.page_count || 22;

  return (
    <div className="min-h-screen bg-[var(--paper)] text-[var(--ink)] flex flex-col font-sans">
      {/* Workspace Header */}
      <header className="h-14 bg-[var(--paper)] border-b border-[var(--rule)] px-6 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <Link href="/" className="text-[var(--ink-soft)] hover:text-[var(--ink)] transition-colors flex items-center gap-1 text-xs font-mono">
            <ArrowLeft size={14} /> LANDING PAGE
          </Link>
          <span className="text-[var(--rule)]">|</span>
          <span className="font-serif text-xl font-bold tracking-tight text-[var(--ink)]">VERITAS</span>
          <span className="font-mono text-xs text-[var(--ink-soft)] px-2 py-0.5 border border-[var(--rule)] rounded-[2px] bg-[var(--paper-2)]">
            Audit Workspace
          </span>
        </div>

        <div className="font-mono text-xs text-[var(--ink-soft)] flex items-center gap-2">
          <span>Mode:</span>
          <span className="text-[var(--accent)] font-semibold">
            {process.env.NEXT_PUBLIC_USE_MOCK === 'true' ? 'DEMO MOCK MODE (NEXT_PUBLIC_USE_MOCK=true)' : 'REAL BACKEND API'}
          </span>
        </div>
      </header>

      {/* Workspace 3-Zone Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Zone A: Documents Rail (280px) */}
        <aside className="w-72 bg-[var(--paper-2)] border-r border-[var(--rule)] p-4 flex flex-col justify-between shrink-0 overflow-y-auto">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-semibold uppercase text-[var(--ink)]">DOCUMENTS</span>
              <span className="font-mono text-xs text-[var(--ink-soft)]">
                {selectedDocIds.length} Selected
              </span>
            </div>

            {/* Upload Zone */}
            <label className="border border-dashed border-[var(--rule)] bg-[#FFFFFF] hover:bg-[var(--surface)] p-4 rounded-[4px] cursor-pointer flex flex-col items-center justify-center text-center transition-colors block">
              <Upload size={18} className="text-[var(--ink-soft)] mb-1" />
              <span className="text-xs font-medium text-[var(--ink)]">Drop a PDF or choose a file</span>
              <span className="text-[10px] text-[var(--ink-soft)] mt-0.5">Max size 50MB</span>
              <input type="file" accept=".pdf" onChange={handleFileUpload} className="hidden" />
            </label>

            {/* Sample Documents Button */}
            <button
              onClick={handleLoadSampleDocs}
              className="w-full bg-[#FFFFFF] border border-[var(--rule)] text-[var(--ink)] text-xs font-medium py-2 rounded-[4px] hover:bg-[var(--surface)] transition-colors flex items-center justify-center gap-1.5"
            >
              <FileText size={14} className="text-[var(--accent)]" /> Try Sample Demo Documents
            </button>

            {/* Scope Header */}
            <div className="font-mono text-[11px] text-[var(--accent)] bg-[#FFFFFF] border border-[var(--rule)] p-2 rounded-[2px]">
              Asking across {selectedDocIds.length} selected document{selectedDocIds.length === 1 ? '' : 's'}
            </div>

            {/* Document List */}
            <div className="space-y-2">
              {documents.map((doc) => {
                const isChecked = selectedDocIds.includes(doc.doc_id);
                return (
                  <div
                    key={doc.doc_id}
                    className={`p-2.5 rounded-[4px] border text-xs transition-all ${
                      isChecked
                        ? 'bg-[#FFFFFF] border-[var(--ink)] shadow-xs'
                        : 'bg-[var(--paper)] border-[var(--rule)] opacity-80'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <label className="flex items-center gap-2 cursor-pointer overflow-hidden flex-1">
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedDocIds([...selectedDocIds, doc.doc_id]);
                            } else {
                              setSelectedDocIds(selectedDocIds.filter((id) => id !== doc.doc_id));
                            }
                          }}
                          className="rounded-[2px] border-[var(--rule)] text-[var(--ink)] focus:ring-[var(--accent)]"
                        />
                        <span className="font-medium text-[var(--ink)] truncate">{doc.document_title}</span>
                      </label>
                      <button
                        onClick={() => {
                          setDocuments(documents.filter((d) => d.doc_id !== doc.doc_id));
                          setSelectedDocIds(selectedDocIds.filter((id) => id !== doc.doc_id));
                        }}
                        className="text-[var(--ink-soft)] hover:text-[var(--bad)] shrink-0"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>

                    <div className="mt-2 flex items-center justify-between font-mono text-[10px] text-[var(--ink-soft)]">
                      <span>{doc.page_count} pages</span>
                      <span className="font-semibold text-[var(--ok)]">{doc.status}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </aside>

        {/* Zone B: Conversation & Tabs (Center) */}
        <main className="flex-1 bg-[var(--paper)] p-6 flex flex-col justify-between overflow-y-auto border-r border-[var(--rule)] min-w-[340px]">
          <div className="space-y-6 max-w-2xl mx-auto w-full">
            {/* Empty State or Query Response */}
            {!currentResponse && !loadingQuery && (
              <div className="border border-[var(--rule)] bg-[#FFFFFF] p-6 rounded-[4px] space-y-4 text-center my-12">
                <h3 className="font-serif text-xl">Audit Research Assistant</h3>
                <p className="text-sm text-[var(--ink-soft)] max-w-md mx-auto">
                  Ask questions across your selected documents. Every claim is linked directly to exact bounding boxes on source PDF pages.
                </p>
                <div className="pt-2 text-left space-y-2">
                  <span className="font-mono text-xs text-[var(--ink-soft)]">SUGGESTED QUESTIONS:</span>
                  <button
                    onClick={() => {
                      setQuestionInput('Compare Line Alpha Actual Output and primary downtime cause between Q2 and Q4.');
                      handleRunQuery('Compare Line Alpha Actual Output and primary downtime cause between Q2 and Q4.');
                    }}
                    className="w-full text-left text-xs bg-[var(--surface)] border border-[var(--rule)] p-2.5 rounded-[4px] hover:bg-[var(--paper-2)] transition-colors font-medium"
                  >
                    1. Compare Line Alpha Actual Output and primary downtime cause between Q2 and Q4.
                  </button>
                  <button
                    onClick={() => {
                      setQuestionInput('What was the primary root cause for downtime in Q2?');
                      handleRunQuery('What was the primary root cause for downtime in Q2?');
                    }}
                    className="w-full text-left text-xs bg-[var(--surface)] border border-[var(--rule)] p-2.5 rounded-[4px] hover:bg-[var(--paper-2)] transition-colors font-medium"
                  >
                    2. What was the primary root cause for downtime in Q2?
                  </button>
                </div>
              </div>
            )}

            {/* Loading State */}
            {loadingQuery && (
              <div className="border border-[var(--rule)] bg-[#FFFFFF] p-6 rounded-[4px] space-y-3">
                <div className="font-mono text-xs text-[var(--accent)] animate-pulse">
                  {queryStage}
                </div>
                <div className="h-4 bg-[var(--paper-2)] rounded-[2px] animate-pulse w-3/4"></div>
                <div className="h-4 bg-[var(--paper-2)] rounded-[2px] animate-pulse w-1/2"></div>
              </div>
            )}

            {/* Response Display */}
            {currentResponse && !loadingQuery && (
              <div className="space-y-6">
                {/* Refusal / Neutral Block */}
                {currentResponse.confidence_badge === 'REFUSAL' ? (
                  <div className="border border-[var(--rule)] bg-[var(--paper-2)] p-5 rounded-[4px] space-y-3">
                    <div className="flex items-center gap-2 text-sm font-semibold text-[var(--warn)]">
                      <AlertCircle size={16} /> No Supported Answer Found
                    </div>
                    <p className="text-xs text-[var(--ink-soft)]">
                      The anti-hallucination evidence gate searched your selected documents and found zero matching evidence units.
                    </p>
                    <div className="text-xs space-y-1 font-mono text-[var(--ink-soft)] border-t border-[var(--rule)] pt-3">
                      <div>Try: Rephrasing your query, adding another document, or selecting additional document checkboxes in the left rail.</div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-[#FFFFFF] border border-[var(--rule)] p-6 rounded-[4px] space-y-4 shadow-xs">
                    {/* Confidence Badge */}
                    <div className="flex items-center justify-between border-b border-[var(--rule)] pb-3">
                      <span className="font-serif text-lg font-semibold">Answer</span>
                      <span className={`font-mono text-xs px-2.5 py-0.5 border rounded-[2px] ${
                        currentResponse.confidence_badge === 'HIGH' ? 'bg-[var(--paper-2)] text-[var(--ok)] border-[var(--ok)]' : 'bg-[var(--paper-2)] text-[var(--warn)] border-[var(--warn)]'
                      }`}>
                        CONFIDENCE: {currentResponse.confidence_badge} ({(currentResponse.confidence * 100).toFixed(0)}%)
                      </span>
                    </div>

                    {/* Answer Text in Serif */}
                    <p className="font-serif text-xl leading-relaxed text-[var(--ink)]">
                      {currentResponse.answer}
                    </p>

                    {/* Citation Chips */}
                    <div className="space-y-2 pt-2 border-t border-[var(--rule)]">
                      <span className="font-mono text-xs text-[var(--ink-soft)]">CITATIONS (CLICK TO VIEW ON PDF):</span>
                      <div className="flex flex-wrap gap-2">
                        {currentResponse.citations.map((c, idx) => (
                          <button
                            key={idx}
                            onClick={() => {
                              setActiveViewerDocId(c.document_id);
                              setActivePageNum(c.page);
                              setHighlightedCitationId(c.element_id);
                            }}
                            className="font-mono text-xs bg-[var(--paper-2)] text-[var(--accent)] border border-[var(--rule)] px-2.5 py-1 rounded-[2px] hover:bg-[var(--mark)] hover:text-[var(--ink)] transition-colors"
                          >
                            {c.document_title} - p.{c.page} - {c.element_type}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Limitation Notes */}
                    {currentResponse.limitations.length > 0 && (
                      <div className="bg-[var(--surface)] border border-[var(--rule)] p-3 rounded-[4px] text-xs text-[var(--warn)] font-mono space-y-1">
                        {currentResponse.limitations.map((lim, i) => (
                          <div key={i}>⚠️ {lim}</div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Tabs Row (Evidence, Calculation, Reasoning) */}
                <div className="border-b border-[var(--rule)] flex gap-6 font-mono text-xs">
                  <button
                    onClick={() => setActiveTab('evidence')}
                    className={`pb-2 font-medium transition-colors border-b-2 ${
                      activeTab === 'evidence' ? 'border-[var(--accent)] text-[var(--accent)]' : 'border-transparent text-[var(--ink-soft)]'
                    }`}
                  >
                    Evidence
                  </button>
                  <button
                    onClick={() => setActiveTab('calculation')}
                    className={`pb-2 font-medium transition-colors border-b-2 ${
                      activeTab === 'calculation' ? 'border-[var(--accent)] text-[var(--accent)]' : 'border-transparent text-[var(--ink-soft)]'
                    }`}
                  >
                    Calculation Trace
                  </button>
                  <button
                    onClick={() => setActiveTab('reasoning')}
                    className={`pb-2 font-medium transition-colors border-b-2 ${
                      activeTab === 'reasoning' ? 'border-[var(--accent)] text-[var(--accent)]' : 'border-transparent text-[var(--ink-soft)]'
                    }`}
                  >
                    Reasoning Plan
                  </button>
                </div>

                {/* Tab Content */}
                {activeTab === 'evidence' && (
                  <div className="space-y-3">
                    {currentResponse.citations.map((c, idx) => (
                      <div key={idx} className="bg-[#FFFFFF] border border-[var(--rule)] p-3.5 rounded-[4px] text-xs font-mono space-y-2">
                        <div className="flex items-center justify-between text-[var(--ink-soft)]">
                          <span>{c.element_id} ({c.element_type})</span>
                          <span>Page {c.page}</span>
                        </div>
                        <p className="font-sans text-xs text-[var(--ink)] bg-[var(--surface)] p-2 rounded-[2px] border border-[var(--rule)]">
                          "{c.snippet}"
                        </p>
                        <button
                          onClick={() => {
                            setActiveViewerDocId(c.document_id);
                            setActivePageNum(c.page);
                            setHighlightedCitationId(c.element_id);
                          }}
                          className="text-[var(--accent)] underline font-sans text-xs"
                        >
                          Show on page →
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {activeTab === 'calculation' && (
                  <div className="bg-[#FFFFFF] border border-[var(--rule)] p-4 rounded-[4px] text-xs font-mono space-y-2">
                    {currentResponse.calculation_trace.length === 0 ? (
                      <p className="text-[var(--ink-soft)]">No calculation was needed for this answer.</p>
                    ) : (
                      currentResponse.calculation_trace.map((step, i) => (
                        <div key={i} className="border-b border-[var(--rule)] pb-2 last:border-0">
                          Step {i + 1}: {JSON.stringify(step)}
                        </div>
                      ))
                    )}
                  </div>
                )}

                {activeTab === 'reasoning' && (
                  <div className="bg-[#FFFFFF] border border-[var(--rule)] p-4 rounded-[4px] text-xs space-y-2">
                    {currentResponse.reasoning_plan.map((step, i) => (
                      <div key={i} className="flex items-start gap-2">
                        <span className="font-mono text-[var(--accent)]">{i + 1}.</span>
                        <span>{step}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Question Input Box */}
          <div className="pt-4 max-w-2xl mx-auto w-full">
            <div className="border border-[var(--rule)] bg-[#FFFFFF] rounded-[4px] p-2 flex items-end gap-2 shadow-xs focus-within:border-[var(--accent)]">
              <textarea
                value={questionInput}
                onChange={(e) => setQuestionInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleRunQuery(questionInput);
                  }
                }}
                placeholder="Ask an audit or compliance question... (Enter to send, Shift+Enter for newline)"
                rows={2}
                className="w-full bg-transparent text-sm text-[var(--ink)] placeholder-[var(--ink-soft)] focus:outline-none resize-none px-2 py-1"
              />
              <button
                onClick={() => handleRunQuery(questionInput)}
                disabled={loadingQuery || !questionInput.trim()}
                className="bg-[var(--ink)] text-[var(--paper)] px-4 py-2 rounded-[4px] text-xs font-medium hover:bg-[var(--accent)] disabled:opacity-40 transition-colors shrink-0 flex items-center gap-1"
              >
                Ask <CornerDownLeft size={12} />
              </button>
            </div>
          </div>
        </main>

        {/* Zone C: PDF Viewer with Canvas / Yellow Bbox Overlays (Right) */}
        <section className="flex-1 bg-[var(--paper-2)] p-4 flex flex-col justify-between overflow-hidden">
          {/* Toolbar */}
          <div className="bg-[#FFFFFF] border border-[var(--rule)] p-3 rounded-[4px] flex items-center justify-between text-xs font-mono shrink-0 mb-4">
            <div className="flex items-center gap-2">
              <FileText size={14} className="text-[var(--accent)]" />
              <span className="font-semibold">{activeDoc?.document_title || activeViewerDocId}</span>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={() => setActivePageNum((p) => Math.max(1, p - 1))}
                disabled={activePageNum <= 1}
                className="p-1 border border-[var(--rule)] rounded-[2px] disabled:opacity-40 hover:bg-[var(--surface)]"
              >
                <ChevronLeft size={14} />
              </button>
              <span>Page {activePageNum} of {maxPages}</span>
              <button
                onClick={() => setActivePageNum((p) => Math.min(maxPages, p + 1))}
                disabled={activePageNum >= maxPages}
                className="p-1 border border-[var(--rule)] rounded-[2px] disabled:opacity-40 hover:bg-[var(--surface)]"
              >
                <ChevronRight size={14} />
              </button>
            </div>

            <div className="flex items-center gap-2">
              <button onClick={() => setZoomLevel((z) => Math.max(50, z - 10))} className="px-2 py-0.5 border border-[var(--rule)] rounded-[2px]">-</button>
              <span>{zoomLevel}%</span>
              <button onClick={() => setZoomLevel((z) => Math.min(150, z + 10))} className="px-2 py-0.5 border border-[var(--rule)] rounded-[2px]">+</button>
            </div>
          </div>

          {/* Canvas PDF Render View */}
          <div className="flex-1 bg-[#FFFFFF] border border-[var(--rule)] rounded-[4px] p-4 overflow-auto flex flex-col items-center justify-center relative">
            <div
              className="relative border border-[var(--rule)] shadow-sm bg-[#FFFFFF] overflow-hidden"
              style={{ width: `${(600 * zoomLevel) / 100}px`, height: `${(780 * zoomLevel) / 100}px` }}
            >
              {/* Render Backend Page Overlay Image */}
              <iframe
                src={`http://localhost:8000/documents/${activeViewerDocId}/pages/${activePageNum}`}
                className="w-full h-full border-0"
                title="PDF Page Viewer"
              />

              {/* Yellow Mark Bounding Box Overlay (#F2D675) */}
              <div
                className={`absolute transition-all border-2 ${
                  highlightedCitationId ? 'border-[var(--ink)] bg-[var(--mark)] opacity-40 animate-pulse' : 'border-[var(--accent)] bg-[var(--mark)] opacity-35'
                }`}
                style={{
                  top: '16%',
                  left: '5%',
                  width: '90%',
                  height: '22%',
                  pointerEvents: 'none'
                }}
              >
                <span className="absolute -top-5 left-1 bg-[var(--ink)] text-[var(--paper)] font-mono text-[9px] px-1 rounded.sm">
                  {highlightedCitationId || 'CITED EVIDENCED BBOX'}
                </span>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
