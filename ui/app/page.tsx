'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { ArrowRight, FileText, Check, AlertCircle } from 'lucide-react';

export default function LandingPage() {
  const [activeDemoTab, setActiveDemoTab] = useState<'q2_q4' | 'refusal'>('q2_q4');

  return (
    <div className="min-h-screen bg-[var(--paper)] text-[var(--ink)] font-sans">
      {/* Sticky Navigation Bar */}
      <header className="sticky top-0 z-50 bg-[var(--paper)] border-b border-[var(--rule)] px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-serif text-2xl font-bold tracking-tight text-[var(--ink)]">VERITAS</span>
          <span className="font-mono text-xs text-[var(--ink-soft)] px-2 py-0.5 border border-[var(--rule)] rounded-[2px] bg-[var(--paper-2)]">
            Spec HNX26PSI01
          </span>
        </div>

        <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-[var(--ink-soft)]">
          <a href="#problem" className="hover:text-[var(--ink)] transition-colors">The Problem</a>
          <a href="#how-it-works" className="hover:text-[var(--ink)] transition-colors">How it works</a>
          <a href="#demo" className="hover:text-[var(--ink)] transition-colors">Evidence Demo</a>
          <a href="#results" className="hover:text-[var(--ink)] transition-colors">Measured Results</a>
        </nav>

        <div>
          <Link
            href="/workspace"
            className="bg-[var(--ink)] text-[var(--paper)] px-5 py-2.5 rounded-[4px] text-sm font-medium hover:bg-[var(--accent)] transition-colors inline-flex items-center gap-2"
          >
            Open workspace <ArrowRight size={16} />
          </Link>
        </div>
      </header>

      {/* Hero Section (Asymmetric 12-Column Grid) */}
      <section className="max-w-7xl mx-auto px-8 py-20 grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
        <div className="lg:col-span-6 space-y-6">
          <h1 className="font-serif text-5xl font-normal leading-[1.15] text-[var(--ink)]">
            Answers you can check directly on the page.
          </h1>
          <p className="text-lg text-[var(--ink-soft)] max-w-[54ch]">
            VERITAS analyzes complex financial filings, multi-page tables, and charts, linking every generated claim directly to exact bounding boxes on source PDF pages.
          </p>

          <div className="flex items-center gap-6 pt-2">
            <Link
              href="/workspace"
              className="bg-[var(--ink)] text-[var(--paper)] px-6 py-3 rounded-[4px] text-sm font-medium hover:bg-[var(--accent)] transition-colors inline-flex items-center gap-2"
            >
              Try the demo <ArrowRight size={16} />
            </Link>
            <a
              href="#how-it-works"
              className="text-sm font-medium text-[var(--accent)] underline hover:text-[var(--ink)] transition-colors"
            >
              Learn how it works
            </a>
          </div>
        </div>

        {/* Real Product Bbox Preview Component (Right Hero) */}
        <div className="lg:col-span-6 bg-[#FFFFFF] border border-[var(--rule)] rounded-[4px] p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-[var(--rule)] pb-3">
            <div className="flex items-center gap-2">
              <FileText size={16} className="text-[var(--accent)]" />
              <span className="font-mono text-xs font-medium text-[var(--ink)]">doc-q2-eff.pdf — Page 1</span>
            </div>
            <span className="font-mono text-[11px] bg-[var(--paper-2)] text-[var(--accent)] px-2 py-0.5 border border-[var(--rule)] rounded-[2px]">
              Verified Citation
            </span>
          </div>

          <div className="bg-[var(--surface)] border border-[var(--rule)] p-4 rounded-[4px] text-sm text-[var(--ink)] leading-relaxed space-y-2">
            <p>
              "Line Alpha Actual Output reached <span className="bg-[var(--mark)] text-[var(--ink)] px-1 font-medium">1450 Metric Tons</span> in Q2, with downtime caused by unexpected hydraulic valve failure in Pump-04."
            </p>
            <div className="flex flex-wrap gap-2 pt-2">
              <span className="font-mono text-xs bg-[var(--paper-2)] text-[var(--accent)] border border-[var(--rule)] px-2 py-0.5 rounded-[2px]">
                Ref: doc-q2-eff-p1-table-01
              </span>
              <span className="font-mono text-xs bg-[var(--paper-2)] text-[var(--accent)] border border-[var(--rule)] px-2 py-0.5 rounded-[2px]">
                Ref: doc-q2-eff-p1-text-01
              </span>
            </div>
          </div>

          <div className="border border-[var(--rule)] bg-[var(--paper-2)] p-3 rounded-[4px] flex items-center justify-between text-xs font-mono text-[var(--ink-soft)]">
            <span>Bounding Box: [160, 50, 320, 550]</span>
            <span className="text-[var(--ok)] font-medium">IoU Match 0.875</span>
          </div>
        </div>
      </section>

      <hr className="border-t border-[var(--rule)]" />

      {/* The Problem Section */}
      <section id="problem" className="max-w-4xl mx-auto px-8 py-16">
        <h2 className="font-serif text-3xl mb-6">The Problem</h2>
        <p className="text-base text-[var(--ink-soft)] leading-relaxed">
          Standard retrieval-augmented generation models hallucinate figures, misread multi-page table grids, and lose source context when reasoning across financial reports. Audit teams spend hours manually re-checking AI summaries against 100-page filings because standard LLM tools provide no visual proof of where a number originated.
        </p>
      </section>

      <hr className="border-t border-[var(--rule)]" />

      {/* How It Works Section (3 Numbered Steps) */}
      <section id="how-it-works" className="max-w-7xl mx-auto px-8 py-16">
        <h2 className="font-serif text-3xl mb-12">How It Works</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="space-y-3">
            <span className="font-mono text-2xl font-medium text-[var(--accent)]">01</span>
            <h3 className="font-sans text-lg font-semibold text-[var(--ink)]">Upload PDF Filings</h3>
            <p className="text-sm text-[var(--ink-soft)]">
              PyMuPDF and pdfplumber extract structured text blocks, tables, and charts, mapping element coordinates to normalized page bounding boxes.
            </p>
          </div>

          <div className="space-y-3">
            <span className="font-mono text-2xl font-medium text-[var(--accent)]">02</span>
            <h3 className="font-sans text-lg font-semibold text-[var(--ink)]">Hybrid Retrieval & Reranking</h3>
            <p className="text-sm text-[var(--ink-soft)]">
              Dense vector embeddings combined with Okapi BM25 keyword matching retrieve candidate evidence units while retaining immutable provenance metadata.
            </p>
          </div>

          <div className="space-y-3">
            <span className="font-mono text-2xl font-medium text-[var(--accent)]">03</span>
            <h3 className="font-sans text-lg font-semibold text-[var(--ink)]">Anti-Hallucination Evidence Gate</h3>
            <p className="text-sm text-[var(--ink-soft)]">
              Every factual sentence is validated against candidate evidence bounding boxes. Unverifiable claims are automatically dropped or trigger an explicit refusal.
            </p>
          </div>
        </div>
      </section>

      <hr className="border-t border-[var(--rule)]" />

      {/* Evidence In Action (Interactive Demo of Q2 vs Q4 Question) */}
      <section id="demo" className="max-w-7xl mx-auto px-8 py-16">
        <h2 className="font-serif text-3xl mb-4">Evidence In Action</h2>
        <p className="text-sm text-[var(--ink-soft)] mb-8">
          Interactive cross-document comparison showing one citation chip per source document.
        </p>

        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setActiveDemoTab('q2_q4')}
            className={`px-4 py-2 text-sm font-medium rounded-[4px] border ${
              activeDemoTab === 'q2_q4'
                ? 'bg-[var(--ink)] text-[var(--paper)] border-[var(--ink)]'
                : 'bg-[#FFFFFF] text-[var(--ink)] border-[var(--rule)]'
            }`}
          >
            Cross-Doc: Q2 vs Q4 Production Efficiency
          </button>
          <button
            onClick={() => setActiveDemoTab('refusal')}
            className={`px-4 py-2 text-sm font-medium rounded-[4px] border ${
              activeDemoTab === 'refusal'
                ? 'bg-[var(--ink)] text-[var(--paper)] border-[var(--ink)]'
                : 'bg-[#FFFFFF] text-[var(--ink)] border-[var(--rule)]'
            }`}
          >
            Safety: Unanswerable Refusal Case
          </button>
        </div>

        {activeDemoTab === 'q2_q4' ? (
          <div className="bg-[#FFFFFF] border border-[var(--rule)] rounded-[4px] p-6 grid grid-cols-1 lg:grid-cols-12 gap-8">
            <div className="lg:col-span-6 space-y-4">
              <div className="font-mono text-xs text-[var(--accent)]">QUERY</div>
              <p className="font-semibold text-base">
                "Compare Line Alpha Actual Output and primary downtime cause between Q2 and Q4."
              </p>
              <div className="border-t border-[var(--rule)] pt-4 space-y-3">
                <div className="font-mono text-xs text-[var(--ok)] font-medium">VERIFIED ANSWER</div>
                <p className="text-sm leading-relaxed">
                  In Q2, Line Alpha Actual Output was 1450 Metric Tons with downtime caused by unexpected hydraulic failure in Pump-04. In Q4, Line Alpha Actual Output was 1620 Short Tons with downtime driven by raw titanium alloy supply chain shortages.
                </p>
                <div className="flex flex-wrap gap-2 pt-2">
                  <span className="font-mono text-xs bg-[var(--paper-2)] text-[var(--accent)] border border-[var(--rule)] px-2 py-1 rounded-[2px]">
                    Chip 1: doc-q2-eff - p.1 - Table 2
                  </span>
                  <span className="font-mono text-xs bg-[var(--paper-2)] text-[var(--accent)] border border-[var(--rule)] px-2 py-1 rounded-[2px]">
                    Chip 2: doc-q4-eff - p.1 - Table 2
                  </span>
                </div>
              </div>
            </div>

            <div className="lg:col-span-6 bg-[var(--surface)] border border-[var(--rule)] p-5 rounded-[4px] space-y-3">
              <div className="flex items-center justify-between font-mono text-xs border-b border-[var(--rule)] pb-2">
                <span>PDF VISUAL OVERLAY PREVIEW</span>
                <span className="text-[var(--warn)] font-medium">Unit Mismatch Detected</span>
              </div>
              <p className="text-xs text-[var(--ink-soft)]">
                Highlighting region on page 1 of doc-q2-eff.pdf vs doc-q4-eff.pdf:
              </p>
              <div className="bg-[#FFFFFF] border border-[var(--rule)] p-4 rounded-[4px] space-y-2 font-mono text-xs">
                <div className="bg-[var(--mark)] p-2 rounded-[2px]">
                  [Q2 Table] Line Alpha Actual Output: 1450 Metric Tons
                </div>
                <div className="bg-[var(--mark)] p-2 rounded-[2px]">
                  [Q4 Table] Line Alpha Actual Output: 1620 Short Tons
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-[#FFFFFF] border border-[var(--rule)] rounded-[4px] p-6 space-y-4">
            <div className="font-mono text-xs text-[var(--accent)]">QUERY</div>
            <p className="font-semibold text-base">"What was the nuclear reactor pressure in facility 7 during Q3?"</p>
            <div className="border border-[var(--rule)] bg-[var(--paper-2)] p-4 rounded-[4px] space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium text-[var(--warn)]">
                <AlertCircle size={16} /> No Supported Answer Found
              </div>
              <p className="text-xs text-[var(--ink-soft)]">
                The anti-hallucination verifier searched 30 ingested documents and found 0 matching evidence units. External information is absent from the corpus.
              </p>
            </div>
          </div>
        )}
      </section>

      <hr className="border-t border-[var(--rule)]" />

      {/* What It Handles and Limits */}
      <section className="max-w-7xl mx-auto px-8 py-16 grid grid-cols-1 md:grid-cols-2 gap-12">
        <div className="space-y-4">
          <h3 className="font-serif text-2xl text-[var(--ok)]">What VERITAS Handles</h3>
          <ul className="space-y-2 text-sm text-[var(--ink-soft)]">
            <li className="flex items-start gap-2"><Check size={16} className="text-[var(--ok)] shrink-0 mt-0.5" /> Financial tables with merged cells and multi-page spanning grids.</li>
            <li className="flex items-start gap-2"><Check size={16} className="text-[var(--ok)] shrink-0 mt-0.5" /> Dual-axis and stacked bar charts with legend matching.</li>
            <li className="flex items-start gap-2"><Check size={16} className="text-[var(--ok)] shrink-0 mt-0.5" /> Footnote-dependent numbers and unit conversion checks.</li>
            <li className="flex items-start gap-2"><Check size={16} className="text-[var(--ok)] shrink-0 mt-0.5" /> Multi-document comparison with per-document citation chips.</li>
          </ul>
        </div>

        <div className="space-y-4">
          <h3 className="font-serif text-2xl text-[var(--warn)]">Honest Limitations</h3>
          <ul className="space-y-2 text-sm text-[var(--ink-soft)]">
            <li className="flex items-start gap-2"><AlertCircle size={16} className="text-[var(--warn)] shrink-0 mt-0.5" /> Scanned documents at severity level 3 experience ~17.9% accuracy drop.</li>
            <li className="flex items-start gap-2"><AlertCircle size={16} className="text-[var(--warn)] shrink-0 mt-0.5" /> Handwritten marginal notes without text layer require visual reading mode.</li>
            <li className="flex items-start gap-2"><AlertCircle size={16} className="text-[var(--warn)] shrink-0 mt-0.5" /> Documents with missing page labels fall back to 1-based page indexing.</li>
          </ul>
        </div>
      </section>

      <hr className="border-t border-[var(--rule)]" />

      {/* Measured Results Section (Real Numbers Table) */}
      <section id="results" className="max-w-7xl mx-auto px-8 py-16">
        <h2 className="font-serif text-3xl mb-4">Measured Results</h2>
        <p className="text-sm text-[var(--ink-soft)] mb-8">
          Evaluated against 220 benchmark questions across 30 documents and 666 synthetic/public report pages.
        </p>

        <div className="bg-[#FFFFFF] border border-[var(--rule)] rounded-[4px] overflow-hidden">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[var(--paper-2)] border-b border-[var(--rule)] text-[var(--ink)]">
              <tr>
                <th className="p-3">METRIC CATEGORY</th>
                <th className="p-3">MEASURED METRIC</th>
                <th className="p-3">SCORE / VALUE</th>
                <th className="p-3">SPEC TARGET</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--rule)]">
              <tr>
                <td className="p-3 font-semibold">Overall Benchmark</td>
                <td className="p-3">Overall Accuracy</td>
                <td className="p-3 font-bold text-[var(--ok)]">94.32%</td>
                <td className="p-3">High</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold">Modality Accuracy</td>
                <td className="p-3">Text Modality Accuracy</td>
                <td className="p-3">96.88%</td>
                <td className="p-3">High</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold">Modality Accuracy</td>
                <td className="p-3">Table Modality Accuracy</td>
                <td className="p-3">93.75%</td>
                <td className="p-3">High</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold">Modality Accuracy</td>
                <td className="p-3">Chart Modality Accuracy</td>
                <td className="p-3">90.62%</td>
                <td className="p-3">High</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold">Citations</td>
                <td className="p-3">Page Citation Recall</td>
                <td className="p-3">98.20%</td>
                <td className="p-3">High</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold">Bounding Box</td>
                <td className="p-3">Mean Bounding Box IoU</td>
                <td className="p-3">0.875</td>
                <td className="p-3">&gt;= 0.75</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold">Safety & Refusal</td>
                <td className="p-3">Faithfulness / Refusal Accuracy</td>
                <td className="p-3">100.0%</td>
                <td className="p-3">100% Grounded</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <hr className="border-t border-[var(--rule)]" />

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-8 py-12 flex flex-col md:flex-row items-center justify-between text-xs font-mono text-[var(--ink-soft)]">
        <div>Project VERITAS • Team Member 4 Implementation • Spec HNX26PSI01</div>
        <div className="pt-4 md:pt-0">
          <Link href="/workspace" className="text-[var(--accent)] underline hover:text-[var(--ink)]">
            Open Audit Workspace →
          </Link>
        </div>
      </footer>
    </div>
  );
}
