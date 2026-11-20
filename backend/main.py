import os
import time
import json
import sqlite3
from datetime import datetime
from typing import Optional
from collections import defaultdict
from fastapi import FastAPI, HTTPException, status, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(
    title="SafeBite API - Sprint 3 (Production Ready)",
    description="Süni intellekt əsaslı qida təhlükəsizliyi analizi, statistika, axtarış/filtr, allergiya profili və rol əsaslı idarəetmə API-si.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "safebite.db")
FRONTEND_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "index.html"))

# --- Rate Limiting (Təhlükəsizlik və Performans) ---
request_log = defaultdict(list)
RATE_LIMIT_MAX = 30
RATE_LIMIT_WINDOW = 60

def check_rate_limit(client_ip: str):
    now = time.time()
    request_log[client_ip] = [t for t in request_log[client_ip] if now - t < RATE_LIMIT_WINDOW]
    if len(request_log[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Çox sayda sorğu göndərildi. Zəhmət olmasa 1 dəqiqə gözləyin.")
    request_log[client_ip].append(now)

# --- Database Setup ---
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS allergies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                severity TEXT NOT NULL DEFAULT 'Yüksək',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                food_name TEXT NOT NULL,
                language TEXT NOT NULL,
                safety_status TEXT NOT NULL,
                halal_status TEXT NOT NULL,
                allergens TEXT NOT NULL,
                summary TEXT NOT NULL,
                user_note TEXT DEFAULT '',
                model_used TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

init_db()

# --- Pydantic Schemas ---
class AnalyzeRequest(BaseModel):
    food_name: str = Field(..., min_length=2, max_length=300, description="Yeməyin adı və ya məhsulun tərkibi")
    language: str = Field(default="Azərbaycan", description="Cavab dili")

class AnalysisUpdate(BaseModel):
    safety_status: str = Field(..., min_length=2, max_length=50, description="Yenilənmiş təhlükəsizlik statusu")
    user_note: Optional[str] = Field(default="", max_length=300, description="İstifadəçinin şəxsi qeydi")

class AllergyCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=60, description="Allergen adı")
    severity: str = Field(default="Yüksək", description="Həssaslıq dərəcəsi: Aşağı, Orta, Yüksək")

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite"
]

# --- Frontend & Health Endpoints ---
@app.get("/", include_in_schema=False)
def serve_frontend():
    if os.path.exists(FRONTEND_PATH):
        return FileResponse(FRONTEND_PATH)
    return {"status": "online", "message": "SafeBite API işləyir. Sənədlər üçün /docs ünvanına keçin."}

@app.get("/api/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "api_key_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "timestamp": datetime.now().isoformat()
    }

# --- Dashboard Statistics ---
@app.get("/api/stats", tags=["Analytics & Stats"])
def get_dashboard_stats():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM analyses").fetchone()["c"]
        allergy_count = conn.execute("SELECT COUNT(*) as c FROM allergies").fetchone()["c"]
        rows = conn.execute("SELECT safety_status FROM analyses").fetchall()

    safe_count = 0
    warn_count = 0
    danger_count = 0
    for r in rows:
        s = (r["safety_status"] or "").lower()
        if "təhlükəli" in s or "danger" in s or "опас" in s:
            danger_count += 1
        elif "təhlükəsiz" in s or "safe" in s or "безопас" in s or "güvenli" in s:
            safe_count += 1
        else:
            warn_count += 1

    return {
        "total_analyses": total,
        "safe_count": safe_count,
        "warning_count": warn_count,
        "danger_count": danger_count,
        "active_allergies": allergy_count
    }

# --- Allergies CRUD ---
@app.get("/api/allergies", tags=["Allergies Profile"])
def list_allergies():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM allergies ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]

@app.post("/api/allergies", status_code=status.HTTP_201_CREATED, tags=["Allergies Profile"])
def add_allergy(payload: AllergyCreate):
    clean_name = payload.name.strip().capitalize()
    try:
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO allergies (name, severity, created_at) VALUES (?, ?, ?)",
                (clean_name, payload.severity, datetime.now().strftime("%Y-%m-%d %H:%M"))
            )
            conn.commit()
            row = conn.execute("SELECT * FROM allergies WHERE id = ?", (cursor.lastrowid,)).fetchone()
            return dict(row)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail=f"'{clean_name}' artıq allergiya profilinizdə mövcuddur.")

@app.delete("/api/allergies/{allergy_id}", tags=["Allergies Profile"])
def delete_allergy(allergy_id: int):
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM allergies WHERE id = ?", (allergy_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Allergiya qeydi tapılmadı.")
        return {"deleted": True, "id": allergy_id}

# --- Food Analysis & History CRUD (with Search & Filter) ---
@app.post("/api/analyze", status_code=status.HTTP_201_CREATED, tags=["Food Analysis (CRUD)"])
def analyze_and_save_food(payload: AnalyzeRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Serverdə GEMINI_API_KEY mühit dəyişəni təyin edilməyib.")

    with get_db() as conn:
        user_allergies = [f"{r['name']} ({r['severity']})" for r in conn.execute("SELECT name, severity FROM allergies").fetchall()]

    allergies_text = ", ".join(user_allergies) if user_allergies else "Xüsusi allergiya qeyd olunmayıb"

    prompt = f"""
    Sən qida təhlükəsizliyi eksperti olan SafeBite süni intellektisən.
    Analiz ediləcək yemək və ya tərkib: "{payload.food_name.strip()}"
    İstifadəçinin şəxsi allergiyaları: {allergies_text}
    Cavab dili: {payload.language}

    Yeməyi analiz et və YALNIZ aşağıdakı struktura uyğun saf JSON qaytar (markdown blokları olmadan):
    {{
      "safety_status": "Təhlükəsiz" və ya "Diqqətli Ol" və ya "Təhlükəli" (seçilən dildə qısa status),
      "halal_status": "Halal" və ya "Şübhəli" və ya "Qeyri-Halal" (seçilən dildə),
      "allergens": "Əsas allergenlər və istifadəçinin allergiyaları ilə kəsişmə (seçilən dildə qısa)",
      "summary": "Tərkib, allergiya riski və halal baxımından 2-3 cümləlik aydın izah (seçilən dildə)"
    }}
    """

    client = genai.Client(api_key=api_key)
    last_error = None
    ai_data = None
    used_model = ""

    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            ai_data = json.loads(response.text)
            used_model = model_name
            break
        except Exception as e:
            last_error = str(e)
            continue

    if not ai_data:
        raise HTTPException(status_code=503, detail=f"Bütün AI modelləri hazırda məşğuldur: {last_error}")

    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyses (food_name, language, safety_status, halal_status, allergens, summary, user_note, model_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.food_name.strip(),
                payload.language,
                ai_data.get("safety_status", "Naməlum"),
                ai_data.get("halal_status", "Naməlum"),
                ai_data.get("allergens", "Məlumat yoxdur"),
                ai_data.get("summary", ""),
                "",
                used_model,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            )
        )
        conn.commit()
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)

@app.get("/api/history", tags=["Food Analysis (CRUD)"])
def get_analysis_history(search: Optional[str] = None, status_filter: Optional[str] = None):
    query = "SELECT * FROM analyses WHERE 1=1"
    params = []

    if search and search.strip():
        query += " AND (LOWER(food_name) LIKE ? OR LOWER(allergens) LIKE ? OR LOWER(summary) LIKE ?)"
        term = f"%{search.strip().lower()}%"
        params.extend([term, term, term])

    if status_filter and status_filter.strip() and status_filter != "ALL":
        query += " AND LOWER(safety_status) LIKE ?"
        params.append(f"%{status_filter.strip().lower()}%")

    query += " ORDER BY id DESC"

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

@app.put("/api/history/{item_id}", tags=["Food Analysis (CRUD)"])
def update_analysis_record(
    item_id: int,
    payload: AnalysisUpdate,
    x_user_role: str = Header(default="user", description="İstifadəçi rolu: user və ya nutritionist")
):
    note_prefix = "[Ekspert Rəyi] " if x_user_role == "nutritionist" and payload.user_note and not payload.user_note.startswith("[Ekspert") else ""
    final_note = f"{note_prefix}{(payload.user_note or '').strip()}"

    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE analyses SET safety_status = ?, user_note = ? WHERE id = ?",
            (payload.safety_status.strip(), final_note, item_id)
        )
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Yenilənəcək analiz qeydi tapılmadı.")
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (item_id,)).fetchone()
        return dict(row)

@app.delete("/api/history/{item_id}", tags=["Food Analysis (CRUD)"])
def delete_analysis_record(
    item_id: int,
    x_user_role: str = Header(default="user", description="İstifadəçi rolu: user və ya nutritionist")
):
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM analyses WHERE id = ?", (item_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Silinəcək analiz qeydi tapılmadı.")
        return {"deleted": True, "id": item_id, "deleted_by_role": x_user_role}
