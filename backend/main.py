import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(
    title="SafeBite API - Sprint 2",
    description="Süni intellekt əsaslı qida təhlükəsizliyi analizi, şəxsi allergiya profili və analiz tarixçəsi (CRUD) API-si.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), "safebite.db")

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

# --- Pydantic Validation Models ---
class AnalyzeRequest(BaseModel):
    food_name: str = Field(..., min_length=2, max_length=120, description="Analiz ediləcək yeməyin adı")
    language: str = Field(default="Azərbaycan", description="Cavab dili")

class AnalysisUpdate(BaseModel):
    safety_status: str = Field(..., min_length=2, max_length=50, description="Yenilənmiş təhlükəsizlik statusu")
    user_note: Optional[str] = Field(default="", max_length=300, description="İstifadəçinin şəxsi qeydi")

class AllergyCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=60, description="Allergen adı (məs: Fıstıq, Qlüten, Süd)")
    severity: str = Field(default="Yüksək", description="Həssaslıq dərəcəsi: Aşağı, Orta, Yüksək")

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite"
]

@app.get("/", tags=["System"])
def root():
    return {"status": "online", "version": "2.0.0", "docs": "/docs"}

# --- ALLERGIES CRUD ---
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

# --- FOOD ANALYSIS & HISTORY CRUD ---
@app.post("/api/analyze", status_code=status.HTTP_201_CREATED, tags=["Food Analysis (CRUD)"])
def analyze_and_save_food(payload: AnalyzeRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Serverdə GEMINI_API_KEY təyin edilməyib.")

    with get_db() as conn:
        user_allergies = [r["name"] for r in conn.execute("SELECT name FROM allergies").fetchall()]

    allergies_text = ", ".join(user_allergies) if user_allergies else "Xüsusi allergiya qeyd olunmayıb"

    prompt = f"""
    Sən qida təhlükəsizliyi eksperti olan SafeBite süni intellektisən.
    Yemək: "{payload.food_name.strip()}"
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
def get_analysis_history():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]

@app.put("/api/history/{item_id}", tags=["Food Analysis (CRUD)"])
def update_analysis_record(item_id: int, payload: AnalysisUpdate):
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE analyses SET safety_status = ?, user_note = ? WHERE id = ?",
            (payload.safety_status.strip(), (payload.user_note or "").strip(), item_id)
        )
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Yenilənəcək analiz qeydi tapılmadı.")
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (item_id,)).fetchone()
        return dict(row)

@app.delete("/api/history/{item_id}", tags=["Food Analysis (CRUD)"])
def delete_analysis_record(item_id: int):
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM analyses WHERE id = ?", (item_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Silinəcək analiz qeydi tapılmadı.")
        return {"deleted": True, "id": item_id}
