from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import re
import json
import httpx
import os

app = FastAPI(title="AI Secure Data Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ────────────────────────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
def root():
    return FileResponse(os.path.join(frontend_path, "index.html"))

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# ── Risk Patterns ─────────────────────────────────────────────────────────────
PATTERNS = {
    "api_key":           {"regex": r"(?i)(api[_-]?key|apikey|access[_-]?key)\s*[=:]\s*([a-zA-Z0-9\-_]{16,})", "risk": "high"},
    "password":          {"regex": r"(?i)(password|passwd|pwd)\s*[=:]\s*\S+",                                   "risk": "critical"},
    "secret_token":      {"regex": r"(?i)(secret|token|bearer)\s*[=:]\s*([a-zA-Z0-9\-_\.]{16,})",              "risk": "critical"},
    "aws_key":           {"regex": r"AKIA[0-9A-Z]{16}",                                                         "risk": "critical"},
    "connection_string": {"regex": r"(?i)(mongodb|mysql|postgresql|redis):\/\/[^\s]+",                          "risk": "critical"},
    "private_key":       {"regex": r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",                          "risk": "critical"},
    "email":             {"regex": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",                       "risk": "low"},
    "phone":             {"regex": r"(\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",                "risk": "low"},
    "ip_address":        {"regex": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                                             "risk": "low"},
    "stack_trace":       {"regex": r"(Traceback \(most recent call last\)|NullPointerException|Exception in thread|at [a-zA-Z]+\.[a-zA-Z]+\()", "risk": "medium"},
}

RISK_SCORE = {"critical": 10, "high": 7, "medium": 4, "low": 1}

# ── Models ────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    input_type: str
    content: str
    api_key: Optional[str] = ""
    options: Optional[dict] = {"mask": True, "block_high_risk": False}

# ── Detection Engine ──────────────────────────────────────────────────────────
def detect_patterns(content: str, mask: bool = True):
    findings = []
    lines = content.split("\n")
    seen = set()

    for line_num, line in enumerate(lines, 1):
        for pattern_name, pattern_data in PATTERNS.items():
            for match in re.finditer(pattern_data["regex"], line):
                value = match.group(0)
                key = (pattern_name, value)
                if key in seen:
                    continue
                seen.add(key)

                if mask and len(value) > 8:
                    masked = value[:4] + "***" + value[-4:]
                else:
                    masked = "***"

                findings.append({
                    "type": pattern_name,
                    "risk": pattern_data["risk"],
                    "value": value,
                    "masked_value": masked,
                    "line": line_num
                })
    return findings

def detect_brute_force(content: str):
    anomalies = []
    failed_attempts = {}
    for line_num, line in enumerate(content.split("\n"), 1):
        if re.search(r"(?i)(failed login|login failed|authentication failed|invalid credentials|unauthorized)", line):
            ip_match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", line)
            ip = ip_match.group(0) if ip_match else "unknown"
            failed_attempts[ip] = failed_attempts.get(ip, 0) + 1

    for ip, count in failed_attempts.items():
        if count >= 3:
            anomalies.append({
                "type": "brute_force_attack",
                "risk": "critical",
                "value": f"{count} failed login attempts from IP {ip}",
                "masked_value": f"{count} failed login attempts from IP {ip}",
                "line": None
            })
    return anomalies

def calculate_risk(findings):
    if not findings:
        return {"score": 0, "level": "safe"}
    scores = [RISK_SCORE.get(f["risk"], 0) for f in findings]
    avg = sum(scores) / len(scores)
    score = min(10, round(avg, 1))
    level = "critical" if score >= 8 else "high" if score >= 6 else "medium" if score >= 3 else "low"
    return {"score": score, "level": level}

# ── AI Insights ───────────────────────────────────────────────────────────────
async def get_ai_insights(content: str, findings: list, api_key: str):
    if not api_key or not api_key.startswith("sk-"):
        return {
            "summary": "No API key provided. Add your Anthropic API key to enable AI insights.",
            "insights": ["Provide an Anthropic API key (starts with sk-ant-...) in the input field."],
            "recommendations": []
        }

    findings_text = json.dumps(
        [{"type": f["type"], "risk": f["risk"], "line": f.get("line")} for f in findings[:15]],
        indent=2
    )

    prompt = f"""You are a cybersecurity analyst. Analyze this data scan result and respond ONLY with a valid JSON object (no markdown, no backticks).

Content preview (first 800 chars):
{content[:800]}

Findings detected:
{findings_text}

Return this exact JSON structure:
{{
  "summary": "one sentence summary of what was found",
  "insights": ["insight 1", "insight 2", "insight 3"],
  "recommendations": ["action 1", "action 2"]
}}"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
        data = response.json()
        text = data["content"][0]["text"].strip()
        text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
        return json.loads(text)
    except Exception as e:
        return {
            "summary": f"AI analysis failed: {str(e)}",
            "insights": ["Check your API key and try again."],
            "recommendations": []
        }

# ── Main Analyze Endpoint ─────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    content = request.content.strip()
    if not content:
        return {"error": "No content provided"}

    mask = request.options.get("mask", True) if request.options else True

    findings = detect_patterns(content, mask=mask)

    if request.input_type in ("log", "text"):
        findings += detect_brute_force(content)

    risk = calculate_risk(findings)

    masked_content = content
    if mask:
        for f in findings:
            if f["value"] != f["masked_value"]:
                masked_content = masked_content.replace(f["value"], f["masked_value"])

    ai = await get_ai_insights(content, findings, request.api_key or "")

    return {
        "input_type": request.input_type,
        "total_findings": len(findings),
        "risk_score": risk["score"],
        "risk_level": risk["level"],
        "findings": findings,
        "masked_content": masked_content,
        "summary": ai.get("summary", ""),
        "insights": ai.get("insights", []),
        "recommendations": ai.get("recommendations", [])
    }

# ── File Upload Endpoint ──────────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content_bytes = await file.read()

    try:
        if filename.endswith(".txt") or filename.endswith(".log"):
            content = content_bytes.decode("utf-8", errors="ignore")

        elif filename.endswith(".pdf"):
            import io
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content_bytes))
            content = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif filename.endswith(".docx"):
            import io
            from docx import Document
            doc = Document(io.BytesIO(content_bytes))
            content = "\n".join(p.text for p in doc.paragraphs)

        else:
            return {"error": "Unsupported file type. Use .txt, .log, .pdf, or .docx"}

        return {"content": content, "filename": file.filename}

    except Exception as e:
        return {"error": f"Could not read file: {str(e)}"}

# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "running", "message": "AI Secure Data Intelligence Platform is live!"}
