# Project VERITAS — Interactive Demo Script & Recorded Fallback Guide

This guide describes how to run the live demonstration for the **"Q2 vs Q4 Production Efficiency"** cross-document audit scenario (HNX26PSI01 Specification Section 17).

---

## 🎬 Live Demo Setup

### 1. Start System Servers
```bash
# Terminal 1: Backend API
$env:PYTHONPATH=".;mdi;src"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Studio Landing Page & Audit Workspace UI
$env:PYTHONPATH=".;mdi;src"
streamlit run ui/studio_app.py --server.port 8501 --server.headless true
```

---

## 📋 Step-by-Step Demo Execution

### Step 1: Open Product Interface
Navigate browser to `http://localhost:8501`.
- Point out the **Studio Landing Page** (`--paper #F4F1EA`, `--ink #14213D`, `--rule #CFC9BA`, `--accent #8C2F1B`).
- Highlight the **0% AI buzzword design**, asymmetric hero section, and real metric evaluation table (94.32% overall accuracy across 220 benchmark questions).

### Step 2: Open Workspace Mode
Click **"Open Audit Workspace →"** or select **"Audit Workspace"** from the top right toggle.

### Step 3: Document Selection
In the multi-select document dropdown, select:
- `doc-q2-eff (Q2 Production Report)`
- `doc-q4-eff (Q4 Production Report)`

### Step 4: Execute Cross-Document Audit Query
Type the exact prompt:
> **"Compare Line Alpha Actual Output and primary downtime cause between Q2 and Q4."**

Click **"Run Audit Query"**.

### Step 5: On-Screen Demonstration Highlights
Point out the following four key UI outputs:

1. **Grounded Answer**:
   - *"In Q2, Line Alpha Actual Output was 1450 Metric Tons with downtime caused by unexpected hydraulic failure in Pump-04. In Q4, Line Alpha Actual Output was 1620 Short Tons with downtime driven by raw titanium alloy supply chain shortages."*

2. **Cross-Document Citation Chips**:
   - `doc-q2-eff-p1-text-01`
   - `doc-q2-eff-p1-table-01`
   - `doc-q4-eff-p1-text-01`
   - `doc-q4-eff-p1-table-01`

3. **Visual Bounding Box Overlay**:
   - Point to the right-hand PDF viewer displaying Page 1.
   - Show the yellow bounding box (`#F2D675`) highlighted over Section 2 Production Output Table.

4. **Plain-Language Limitation Warning**:
   - Highlight the limitation warning box:
     > *"⚠️ Limitation Warning: Notice: doc-q2-eff measures output in Metric Tons whereas doc-q4-eff measures output in Short Tons."*

---

## 🛡️ Fallback Recorded Mock Mode

If live inference or network access is unavailable during presentation:

1. Enable Recorded Mock Mode:
   ```bash
   $env:USE_MOCK="true"
   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. In Mock Mode, queries to the API immediately return realistic pre-recorded demo responses with exact bounding box citations, calculation traces, and unit limitation warnings without external LLM API dependency.
