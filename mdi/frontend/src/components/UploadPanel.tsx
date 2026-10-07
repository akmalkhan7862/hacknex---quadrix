import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { api } from '../services/api';

interface UploadPanelProps {
  onUploadSuccess: () => void;
}

export const UploadPanel: React.FC<UploadPanelProps> = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const files = Array.from(e.target.files);
    
    setUploading(true);
    setStatusMessage('Uploading and processing documents...');
    
    try {
      await api.uploadDocuments(files);
      setStatusMessage('Upload complete!');
      onUploadSuccess();
    } catch (err: any) {
      console.error(err);
      setStatusMessage(`Upload failed: ${err.message || 'Error uploading files'}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-slate-800/80 border border-slate-700/60 rounded-xl p-4 shadow-lg backdrop-blur">
      <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
        <Upload size={16} className="text-indigo-400" />
        Upload Documents
      </h3>

      <label className="flex flex-col items-center justify-center w-full h-28 border-2 border-dashed border-slate-600 hover:border-indigo-500 rounded-lg cursor-pointer bg-slate-900/50 hover:bg-slate-900 transition-all">
        <div className="flex flex-col items-center justify-center pt-2 pb-3">
          {uploading ? (
            <Loader2 className="w-8 h-8 text-indigo-400 animate-spin mb-1" />
          ) : (
            <FileText className="w-8 h-8 text-slate-400 mb-1" />
          )}
          <p className="text-xs text-slate-300 font-medium">
            {uploading ? 'Processing PDFs...' : 'Click to select or drag PDF files'}
          </p>
          <p className="text-[10px] text-slate-500 mt-1">Multi-page PDF Supported</p>
        </div>
        <input
          type="file"
          multiple
          accept=".pdf"
          className="hidden"
          onChange={handleFileChange}
          disabled={uploading}
        />
      </label>

      {statusMessage && (
        <div className="mt-3 flex items-center gap-2 text-xs text-slate-300 bg-slate-900/80 p-2 rounded border border-slate-700">
          {uploading ? (
            <Loader2 size={14} className="text-indigo-400 animate-spin" />
          ) : statusMessage.includes('failed') ? (
            <AlertCircle size={14} className="text-rose-400" />
          ) : (
            <CheckCircle size={14} className="text-emerald-400" />
          )}
          <span>{statusMessage}</span>
        </div>
      )}
    </div>
  );
};
