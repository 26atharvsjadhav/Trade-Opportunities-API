import os
import re
import jwt
import json
import time
import uuid
import httpx
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict
from xml.etree import ElementTree as ET

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    Response,
    status,
    Query,
    Path,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field


# Config & Logging
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGO = "HS256"
TOKEN_TTL_MIN = int(os.getenv("TOKEN_TTL_MIN", "1440"))  # 24h default

RATE_LIMIT_COUNT = int(os.getenv("RATE_LIMIT_COUNT", "10"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "300"))  # 5 minutes

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

DEMO_USERNAME = os.getenv("DEMO_USERNAME", "demo")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demo123")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("trade-opps-api")


# Utilities: JWT Auth & Sessions

class AuthPayload(BaseModel):
    sub: str
    exp: int

class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

http_bearer = HTTPBearer(auto_error=False)


def create_jwt(sub: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_TTL_MIN)
    payload = {"sub": sub, "exp": int(exp.timestamp())}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    return token


def decode_jwt(token: str) -> AuthPayload:
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return AuthPayload(**data)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_identity(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
) -> str:
    """Returns a stable identity for rate-limiting and session tracking.
    - If JWT provided: use its `sub`.
    - Otherwise: fall back to client IP.
    """
    if creds and creds.scheme.lower() == "bearer":
        token = creds.credentials
        payload = decode_jwt(token)
        request.state.is_guest = False
        return payload.sub
    request.state.is_guest = True
    ip = request.client.host if request.client else "0.0.0.0"
    return f"guest:{ip}"


# Rate Limiter (Token Bucket - deque window)

class WindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self.events: Dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> Tuple[bool, int]:
        now = time.time()
        dq = self.events[key]
        # purge old
        while dq and now - dq[0] > self.window:
            dq.popleft()
        if len(dq) < self.limit:
            dq.append(now)
            remaining = self.limit - len(dq)
            return True, remaining
        else:
            remaining = 0
            return False, remaining

rate_limiter = WindowRateLimiter(RATE_LIMIT_COUNT, RATE_LIMIT_WINDOW_SEC)


async def rate_limit_dep(request: Request, identity: str = Depends(get_current_identity)):
    allowed, remaining = rate_limiter.allow(identity)
    if not allowed:
        reset_in = RATE_LIMIT_WINDOW_SEC  # simple estimate
        headers = {
            "X-RateLimit-Limit": str(RATE_LIMIT_COUNT),
            "X-RateLimit-Remaining": str(remaining),
            "Retry-After": str(reset_in),
        }
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers=headers,
        )
    request.state.rate_remaining = remaining



SESSIONS: Dict[str, Dict[str, Any]] = {}

async def ensure_session(request: Request, response: Response):
    # For guests only, attach a simple transient session id
    if getattr(request.state, "is_guest", False):
        sid = request.headers.get("X-Session-Id")
        if not sid:
            sid = str(uuid.uuid4())
        if sid not in SESSIONS:
            SESSIONS[sid] = {"created_at": time.time(), "count": 0}
        SESSIONS[sid]["count"] += 1
        response.headers["X-Session-Id"] = sid


SECTOR_PATTERN = re.compile(r"^[a-zA-Z\s\-&]{2,40}$")


def validate_sector(name: str) -> str:
    name = name.strip().lower()
    if not SECTOR_PATTERN.match(name):
        raise HTTPException(status_code=400, detail="Invalid sector name")
    return name


# Data Collection (Search/RSS)

class WebCollector:
    """Collect recent market/news signals for a sector from public endpoints.
    Uses:
      - Google News RSS (no key)
      - DuckDuckGo Instant Answer (limited but keyless)
    """

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def google_news_rss(self, query: str, num: int = 8) -> List[Dict[str, str]]:
        # Example: https://news.google.com/rss/search?q=pharmaceuticals%20India
        url = "https://news.google.com/rss/search"
        params = {"q": f"{query} India", "hl": "en-IN"}
        r = await self.client.get(url, params=params, timeout=15)
        r.raise_for_status()
        items: List[Dict[str, str]] = []
        root = ET.fromstring(r.text)
        for item in root.iterfind("channel/item"):
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")
            if title_el is None or link_el is None:
                continue
            items.append(
                {
                    "title": (title_el.text or "").strip(),
                    "link": (link_el.text or "").strip(),
                    "published": (pub_el.text or "").strip(),
                }
            )
            if len(items) >= num:
                break
        return items

    async def duckduckgo_instant(self, query: str) -> Dict[str, Any]:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        r = await self.client.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    async def collect(self, sector: str, n: int = 8) -> Dict[str, Any]:
        q = f"{sector} sector market India 2025"
        news = await self.google_news_rss(q, num=n)
        ddg = await self.duckduckgo_instant(q)
        return {"news": news, "ddg": ddg}


# Analysis with Gemini
async def analyze_with_gemini(context: Dict[str, Any], sector: str) -> str:
    """Returns a Markdown report."""
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            system = (
                "You are a senior India-focused equity research analyst creating concise, actionable, "
                "risk-aware sector briefs. Include concrete data points, named companies, and cite links."
            )
            prompt = (
                f"Create a structured MARKET OPPORTUNITY brief for the '{sector}' sector in India.\n"
                f"Use only the provided context. If unsure, say so explicitly.\n\n"
                f"Return STRICTLY valid Markdown with these sections:\n"
                "# {Sector} — Trade Opportunities (India)\n"
                "Date: <today in YYYY-MM-DD IST>\n\n"
                "## Snapshot\n- Why now (3 bullets)\n- Tailwinds/Headwinds (bullets)\n\n"
                "## Sub-segments & Themes\n- 3–5 themes with 1–2 lines each\n\n"
                "## Notable Companies & Signals\n- Bulleted list with brief rationale\n\n"
                "## Risks & What to Monitor\n- Bullets\n\n"
                "## Data/News Sources\n- List of clickable links used\n\n"
                "## Bottom Line\n- 2–3 lines\n"
            )

            today_ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d")
            news_lines = []
            for it in context.get("news", [])[:12]:
                news_lines.append(f"- {it.get('title','').strip()} ({it.get('link','')}) | {it.get('published','')}")
            ddg_abstract = context.get("ddg", {}).get("AbstractText") or ""

            content = (
                f"SYSTEM:\n{system}\n\n"
                f"TODAY_IST: {today_ist}\n\n"
                f"CONTEXT_NEWS:\n" + "\n".join(news_lines) + "\n\n"
                f"CONTEXT_DDG_ABSTRACT:\n{ddg_abstract}\n\n"
                f"USER_PROMPT:\n{prompt}\n"
            )

            resp = await model.generate_content_async(content)
            text = resp.text or ""
            if not text.strip():
                text = f"# {sector.title()} — Trade Opportunities (India)\n\n_No content generated._"
            return text
        except Exception as e:
            logger.exception("Gemini analysis failed, falling back: %s", e)
    return render_fallback_report(context, sector)


def render_fallback_report(context: Dict[str, Any], sector: str) -> str:
    today_ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d")
    links = [it.get("link", "") for it in context.get("news", []) if it.get("link")]
    titles = [it.get("title", "") for it in context.get("news", [])]
    ddg_abs = context.get("ddg", {}).get("AbstractText") or ""

    snapshot_bullets = [
        "Multiple fresh headlines and policy updates indicate active capital flows.",
        "Domestic demand and export dynamics show near-term catalysts.",
        "Monitor FX, rates, and regulatory circulars impacting margins and pricing.",
    ]
    themes = [
        "Formalization & compliance-driven consolidation.",
        "Digital distribution and supply-chain visibility platforms.",
        "Export substitution and PLI-linked capacity expansions.",
        "Sustainability-led capex and energy efficiency.",
        "AI/automation in operations and customer acquisition.",
    ]
    companies = titles[:5]

    md = []
    md.append(f"# {sector.title()} — Trade Opportunities (India)")
    md.append(f"Date: {today_ist}")
    md.append("")

    md.append("## 1. Sector Snapshot")
    for b in snapshot_bullets:
        md.append(f"- {b}")
    md.append("")

    md.append("### Tailwinds")
    md.append("- Strong demand and supportive policy initiatives.")
    md.append("- Capacity expansion and formalization trends.")
    md.append("")

    md.append("### Headwinds")
    md.append("- Regulatory uncertainty; global macro pressures.")
    md.append("- Input cost fluctuations.")
    md.append("")

    md.append("## 2. Market Themes & Sub-Segments")
    for t in themes:
        md.append(f"1. {t}")
    md.append("")

    md.append("## 3. Notable Companies & Market Signals")
    for c in companies:
        md.append(f"- {c}")
    md.append("")

    md.append("## 4. Risks & Monitoring Points")
    md.append("- Policy/tariff changes.")
    md.append("- Demand volatility.")
    md.append("- Competitive pressures.")
    md.append("")

    md.append("## 5. Recent News & Data Sources")
    for u in links[:10]:
        md.append(f"- {u}")
    md.append("")

    md.append("## 6. Bottom Line")
    md.append("- Near-term opportunities exist; focus on firms with pricing power.")
    md.append("- Re-evaluate as new quarterly data and policies emerge.")

    return "\n".join(md)


# FastAPI App & Routes
app = FastAPI(title="Trade Opportunities API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


@app.post("/auth/login", response_model=LoginResponse, tags=["auth"])  # simple demo login
async def login(body: LoginRequest):
    if body.username != DEMO_USERNAME or body.password != DEMO_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_jwt(sub=body.username)
    return LoginResponse(access_token=token, expires_in=TOKEN_TTL_MIN * 60)


@app.get("/health", tags=["system"])  # health check
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/analyze/{sector}", response_class=Response, tags=["analyze"], responses={200: {"content": {"text/markdown": {}}}})
async def analyze_sector(
    request: Request,
    response: Response,
    sector: str = Path(..., description="Sector name, e.g., pharmaceuticals, technology, agriculture"),
    n: int = Query(8, ge=3, le=15, description="Number of recent items to fetch"),
    _=Depends(rate_limit_dep),
    identity: str = Depends(get_current_identity),
):
    """Main endpoint: returns a structured Markdown sector brief."""
    sector = validate_sector(sector)

    # Collect data
    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": "trade-opps-api/1.0"}) as client:
        collector = WebCollector(client)
        try:
            context = await collector.collect(sector, n)
        except httpx.HTTPError as e:
            logger.exception("Data collection failed: %s", e)
            raise HTTPException(status_code=502, detail="Upstream fetch failed")

    # Analyze
    report_md = await analyze_with_gemini(context, sector)

    # Session & headers
    await ensure_session(request, response)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_COUNT)
    response.headers["X-RateLimit-Remaining"] = str(getattr(request.state, "rate_remaining", 0))

    # Return as text/markdown; also propose a filename
    filename = f"{sector.replace(' ', '-')}-{datetime.now().strftime('%Y%m%d')}.md"
    response.headers["Content-Disposition"] = f"inline; filename={filename}"
    return Response(content=report_md, media_type="text/markdown; charset=utf-8")


# Error Handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTPException %s: %s", exc.status_code, exc.detail)
    return Response(
        content=json.dumps({"error": exc.detail}),
        status_code=exc.status_code,
        media_type="application/json",
    )


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # Basic security headers
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
