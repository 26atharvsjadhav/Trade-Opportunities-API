"""
Trade Opportunities API — FastAPI 
===================================================

README (Setup & Usage)
----------------------
1) Requirements (Python 3.10+ recommended)
   pip install fastapi uvicorn httpx pydantic PyJWT python-multipart
   pip install google-generativeai

2) Environment variables
   - GEMINI_API_KEY=<your_google_generative_ai_api_key> 
   - JWT_SECRET=<any-random-secret-string>                
   - DEMO_USERNAME=demo                                   
   - DEMO_PASSWORD=demo123                                

3) Run the server
   uvicorn app:app --reload --port 8000

4) Auth (simple JWT demo)
   - POST /auth/login   {"username":"demo","password":"demo123"}
   - Copy the token from the response and send as header:
       Authorization: Bearer <token>

5) Analyze endpoint (core requirement)
   - GET /analyze/{sector}
   - Example:
       curl -H "Authorization: Bearer <token>" \
            "http://localhost:8000/analyze/pharmaceuticals?n=8"
   - Response: Markdown text suitable to save as a .md file.

6) Rate limiting & sessions
   - Per-user (JWT subject) or per-IP (guest) token-bucket: default 10 requests/5 minutes.
   - A simple session id is issued via the `X-Session-Id` response header for guests.

7) Notes
   - In-memory storage only (no DB), as required.
   - Clean separation of concerns within one file: search -> analysis -> API layer.
   - Fallback analysis (no Gemini): a deterministic heuristic summarizer is used.

"""