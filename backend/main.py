import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

app = FastAPI(
    title="SafeBite API",
    description="AI Travel Menu & Ingredient Analyzer MVP API",
    version="1.0.0"
)

# Frontend-in backend-ə qoşulması üçün CORS icazəsi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API konfiqurasiyası (Açarını bura və ya mühit dəyişəninə yaz)
API_KEY = os.getenv("GEMINI_API_KEY", "BURAYA_GEMINI_API_KEY_YAZ")
client = genai.Client(api_key=API_KEY)

class DishRequest(BaseModel):
    dish_name: str
    target_language: str = "Azerbaijani"

class DishResponse(BaseModel):
    original_name: str
    analysis: str

@app.get("/", tags=["Health"])
def read_root():
    return {"status": "ok", "message": "SafeBite API is running"}

@app.post("/api/analyze", response_model=DishResponse, tags=["Menu Analyzer"])
def analyze_dish(request: DishRequest):
    if not request.dish_name.strip():
        raise HTTPException(status_code=400, detail="Dish name cannot be empty")
    
    prompt = f"""
    Analyze this dish or food item: "{request.dish_name}".
    Respond in {request.target_language}.
    Provide a clear, structured breakdown with:
    1. Translated Name & Brief Description
    2. Likely Ingredients
    3. Dietary & Safety Alerts (Halal status / Pork check, Vegan status, Common Allergens, Spice level).
    Keep it concise and easy for a traveler to read.
    """
    
    try:
        for m_name in ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]:
            try:
                response = client.models.generate_content(
                    model=m_name,
            contents=prompt,
                )
                break
            except Exception as _e:
                if m_name == "gemini-3.1-flash-lite": raise _e
        return DishResponse(
            original_name=request.dish_name,
            analysis=response.text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
