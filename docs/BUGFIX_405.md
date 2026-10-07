# Root Cause & Bug Fix Analysis: HTTP 405 on POST /documents

## Observed Bug
When uploading a PDF file from UI clients or fetch requests to `POST /documents`, an HTTP 405 Method Not Allowed error occurred.

## Root Cause Analysis
1. **Trailing Slash Mismatch**: The backend router in `api/main.py` registered `@app.post("/documents")` without trailing slash, while UI fetch requests directed to `/documents/` (with trailing slash) triggered FastAPI's automatic HTTP 307 redirect, which converts `POST` requests or causes CORS preflight method rejection.
2. **OPTIONS Preflight Mismatch**: Browsers sending `OPTIONS /documents` preflight headers with `Content-Type: multipart/form-data` were rejected if routing rules redirected preflight OPTIONS requests.

## Resolution
1. Registered dual route annotations in `api/main.py`: `@app.post("/documents")` and `@app.post("/documents/")`.
2. Verified CORS middleware configuration with `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`.
3. Added automated Pytest + `httpx` test in `tests/test_api.py` targeting `POST /documents` and `POST /documents/` asserting status 200/202 with `{document_id, job_id}` payload.
