import './globals.css';
import React from 'react';

export const metadata = {
  title: 'VERITAS — Multimodal Document Intelligence',
  description: 'Document question-answering tool that returns answers with cited evidence highlighted on the source PDF.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[var(--paper)] text-[var(--ink)] antialiased min-h-screen">
        {children}
      </body>
    </html>
  );
}
