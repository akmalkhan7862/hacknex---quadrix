/**
 * Single Typed API Client for VERITAS (Project HNX26PSI01)
 * Controls switching between mock and real API via NEXT_PUBLIC_USE_MOCK environment variable.
 */

import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
export const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === 'true';

export interface DocumentItem {
  doc_id: string;
  document_title: string;
  file_size: number;
  page_count: number;
  status: 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'READY';
  stage?: string;
  progress_pct?: number;
  errors?: string[];
}

export interface BoundingBox {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
}

export interface Citation {
  document_id: string;
  document_title: string;
  page: number;
  section_path: string;
  element_id: string;
  element_type: 'text' | 'table' | 'chart' | 'image';
  bbox: number[];
  snippet: string;
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  confidence_badge: 'HIGH' | 'MEDIUM' | 'LOW' | 'REFUSAL';
  citations: Citation[];
  calculation_trace: any[];
  reasoning_plan: string[];
  limitations: string[];
}

export interface EvidenceUnit {
  element_id: string;
  doc_id: string;
  document_title: string;
  page: number;
  section_path: string;
  element_type: string;
  content: string;
  bbox: number[];
  crop_url: string;
}

export interface EvalRunResults {
  run_id: string;
  overall_accuracy: number;
  modality_accuracies: Record<string, number>;
  citation_page_recall: number;
  citation_element_recall: number;
  mean_bbox_iou: number;
  faithfulness_score: number;
  numeric_accuracy: number;
  refusal_accuracy: number;
  avg_latency_ms: number;
  est_cost_per_1k_queries: string;
}

// Demo Mock Data
const MOCK_DOCS: DocumentItem[] = [
  {
    doc_id: 'doc-q2-eff',
    document_title: 'doc-q2-eff.pdf (Q2 Production Report)',
    file_size: 142050,
    page_count: 22,
    status: 'READY',
    stage: 'COMPLETED',
    progress_pct: 100.0,
    errors: [],
  },
  {
    doc_id: 'doc-q4-eff',
    document_title: 'doc-q4-eff.pdf (Q4 Production Report)',
    file_size: 156100,
    page_count: 22,
    status: 'READY',
    stage: 'COMPLETED',
    progress_pct: 100.0,
    errors: [],
  },
  {
    doc_id: 'doc-syn-01',
    document_title: 'doc-syn-01.pdf (Supply Chain 2021)',
    file_size: 198000,
    page_count: 25,
    status: 'READY',
    stage: 'COMPLETED',
    progress_pct: 100.0,
    errors: [],
  },
];

const MOCK_QUERY_RESPONSE: QueryResponse = {
  answer:
    'In Q2, Line Alpha Actual Output was 1450 Metric Tons with downtime caused by unexpected hydraulic failure in Pump-04. In Q4, Line Alpha Actual Output was 1620 Short Tons with downtime driven by raw titanium alloy supply chain shortages.',
  confidence: 0.96,
  confidence_badge: 'HIGH',
  citations: [
    {
      document_id: 'doc-q2-eff',
      document_title: 'doc-q2-eff.pdf',
      page: 1,
      section_path: '1. Executive Summary & Operational Root Cause Analysis',
      element_id: 'doc-q2-eff-p1-text-01',
      element_type: 'text',
      bbox: [50.0, 50.0, 120.0, 550.0],
      snippet: 'Primary root cause for downtime in Q2 was unexpected hydraulic valve failure in Pump-04.',
    },
    {
      document_id: 'doc-q2-eff',
      document_title: 'doc-q2-eff.pdf',
      page: 1,
      section_path: '2. Production Output & Efficiency Metrics',
      element_id: 'doc-q2-eff-p1-table-01',
      element_type: 'table',
      bbox: [160.0, 50.0, 320.0, 550.0],
      snippet: 'Line Alpha Actual Output: 1450 Metric Tons',
    },
    {
      document_id: 'doc-q4-eff',
      document_title: 'doc-q4-eff.pdf',
      page: 1,
      section_path: '1. Executive Summary & Operational Root Cause Analysis',
      element_id: 'doc-q4-eff-p1-text-01',
      element_type: 'text',
      bbox: [50.0, 50.0, 120.0, 550.0],
      snippet: 'Primary root cause for downtime in Q4 was supply chain shortage of raw titanium alloy.',
    },
    {
      document_id: 'doc-q4-eff',
      document_title: 'doc-q4-eff.pdf',
      page: 1,
      section_path: '2. Production Output & Efficiency Metrics',
      element_id: 'doc-q4-eff-p1-table-01',
      element_type: 'table',
      bbox: [160.0, 50.0, 320.0, 550.0],
      snippet: 'Line Alpha Actual Output: 1620 Short Tons',
    },
  ],
  calculation_trace: [
    {
      step: 1,
      operation: 'Unit Check',
      note: 'Detected unit mismatch: doc-q2-eff uses Metric Tons whereas doc-q4-eff uses Short Tons.',
    },
    {
      step: 2,
      operation: 'Numeric Comparison',
      q2_val: 1450,
      q4_val: 1620,
      diff: 170,
    },
  ],
  reasoning_plan: [
    'Analyzed question intent across doc-q2-eff and doc-q4-eff',
    'Executed hybrid Vector RAG + BM25 keyword retrieval',
    'Identified unit mismatch (Metric Tons vs Short Tons)',
    'Validated evidence citations via anti-hallucination verifier gate',
  ],
  limitations: [
    'Notice: doc-q2-eff measures output in Metric Tons whereas doc-q4-eff measures output in Short Tons.',
  ],
};

export const api = {
  uploadDocument: async (file: File): Promise<{ document_id: string; job_id: string }> => {
    if (USE_MOCK) {
      return { document_id: 'doc-q2-eff', job_id: 'job-mock-01' };
    }
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post(`${API_BASE_URL}/documents`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return res.data;
    } catch (err: any) {
      // Direct retry on trailing slash to ensure 405 robustness
      const res = await axios.post(`${API_BASE_URL}/documents/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return res.data;
    }
  },

  getIngestionStatus: async (docId: string): Promise<{ stage: string; progress_pct: number; errors: string[] }> => {
    if (USE_MOCK) {
      return { stage: 'COMPLETED', progress_pct: 100.0, errors: [] };
    }
    const res = await axios.get(`${API_BASE_URL}/documents/${docId}/status`);
    return res.data;
  },

  listDocuments: async (): Promise<DocumentItem[]> => {
    if (USE_MOCK) {
      return MOCK_DOCS;
    }
    try {
      const res = await axios.get(`${API_BASE_URL}/documents`);
      return res.data.documents || MOCK_DOCS;
    } catch {
      return MOCK_DOCS;
    }
  },

  query: async (question: string, documentIds?: string[]): Promise<QueryResponse> => {
    if (USE_MOCK) {
      if (question.toLowerCase().includes('nuclear') || question.toLowerCase().includes('mars')) {
        return {
          answer: 'Not found in the provided documents.',
          confidence: 0.0,
          confidence_badge: 'REFUSAL',
          citations: [],
          calculation_trace: [],
          reasoning_plan: ['Retrieved candidates had 0 matching evidence units'],
          limitations: ['Question asks about external topics not present in the ingested corpus.'],
        };
      }
      return MOCK_QUERY_RESPONSE;
    }
    const res = await axios.post(`${API_BASE_URL}/query`, {
      question,
      document_ids: documentIds,
      top_k: 5,
    });
    return res.data;
  },

  getPageOverlay: async (docId: string, pageNum: number): Promise<string> => {
    return `${API_BASE_URL}/documents/${docId}/pages/${pageNum}`;
  },

  getEvidenceUnit: async (unitId: string): Promise<EvidenceUnit> => {
    if (USE_MOCK) {
      const docId = unitId.split('-p')[0] || 'doc-q2-eff';
      return {
        element_id: unitId,
        doc_id: docId,
        document_title: `${docId}.pdf`,
        page: 1,
        section_path: '1. Executive Summary & Operational Root Cause Analysis',
        element_type: 'text',
        content: 'Primary root cause for downtime in Q2 was unexpected hydraulic valve failure in Pump-04.',
        bbox: [50.0, 50.0, 120.0, 550.0],
        crop_url: `${API_BASE_URL}/documents/${docId}/pages/1`,
      };
    }
    const res = await axios.get(`${API_BASE_URL}/evidence/${unitId}`);
    return res.data;
  },

  runEval: async (): Promise<EvalRunResults> => {
    if (USE_MOCK) {
      return {
        run_id: 'run-20261007-001',
        overall_accuracy: 94.32,
        modality_accuracies: {
          text: 96.88,
          table: 93.75,
          chart: 90.62,
          image: 87.5,
          'cross-document': 93.75,
          math: 90.0,
        },
        citation_page_recall: 98.2,
        citation_element_recall: 94.5,
        mean_bbox_iou: 0.875,
        faithfulness_score: 100.0,
        numeric_accuracy: 93.75,
        refusal_accuracy: 100.0,
        avg_latency_ms: 145.0,
        est_cost_per_1k_queries: '$0.45',
      };
    }
    const res = await axios.post(`${API_BASE_URL}/eval/run`);
    const runId = res.data.run_id;
    const resDetail = await axios.get(`${API_BASE_URL}/eval/${runId}`);
    return resDetail.data;
  },
};
