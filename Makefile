# Project VERITAS Makefile

.PHONY: data api ui eval test help

help:
	@echo "Project VERITAS Build Commands:"
	@echo "  make data    Build synthetic dataset, degraded variants, and questions"
	@echo "  make api     Run FastAPI backend API (port 8000)"
	@echo "  make ui      Run Next.js Studio UI App in ui/ (port 3000)"
	@echo "  make eval    Run evaluation suite and output metrics table"
	@echo "  make test    Run pytest test suite"

data:
	@echo "[VERITAS] Building synthetic dataset, degraded variants, and benchmark questions..."
	python scripts/build_dataset.py --seed 42
	python scripts/degrade_docs.py --seed 42
	python scripts/generate_questions.py --seed 42

api:
	@echo "[VERITAS] Starting FastAPI API Backend Server..."
	python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

ui:
	@echo "[VERITAS] Starting Next.js Studio UI App in ui/..."
	cd ui && cmd /c "set NEXT_PUBLIC_USE_MOCK=true && npm run dev"

eval:
	@echo "[VERITAS] Running Evaluation Metrics & Ablation Suite..."
	python eval/run_eval.py

test:
	@echo "[VERITAS] Running Pytest Unit & Integration Tests..."
	python -m pytest tests/test_api.py tests/test_e2e_playwright.py mdi/tests/test_all.py
