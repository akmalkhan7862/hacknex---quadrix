import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export interface DocumentItem {
  doc_id: string;
  document_title: string;
  file_size: number;
  page_count: number;
  status: 'UPLOADED' | 'PROCESSING' | 'PROCESSED' | 'FAILED';
  created_at?: string;
  error_message?: string;
}

export interface Claim {
  text: string;
  evidence_ids: string[];
}

export interface EvidenceSource {
  document: string;
  page: number;
  section: string;
  content_type: 'text' | 'table';
  element_id: string;
  snippet?: string;
}

export interface QueryResponse {
  answer: string;
  claims: Claim[];
  sources: EvidenceSource[];
  calculations: any[];
}

export const api = {
  uploadDocuments: async (files: File[]) => {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    const res = await axios.post(`${API_BASE_URL}/documents/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  listDocuments: async (): Promise<{ documents: DocumentItem[] }> => {
    const res = await axios.get(`${API_BASE_URL}/documents`);
    return res.data;
  },

  getDocumentDetails: async (docId: string) => {
    const res = await axios.get(`${API_BASE_URL}/documents/${docId}`);
    return res.data;
  },

  query: async (question: string): Promise<QueryResponse> => {
    const res = await axios.post(`${API_BASE_URL}/query`, { question, top_k: 5 });
    return res.data;
  },

  getEvidence: async (elementId: string) => {
    const res = await axios.get(`${API_BASE_URL}/evidence/${elementId}`);
    return res.data;
  },
};
