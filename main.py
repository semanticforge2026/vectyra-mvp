from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import requests
import os

app = FastAPI(title="Vectyra AI API", version="1.0")

# ДОЗВОЛЯЄМО ФРОНТЕНДУ СПІЛКУВАТИСЯ З БЕКЕНДОМ (CORS)
# Це дуже важливо, щоб браузер не блокував запити!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В ідеалі тут має бути посилання на твій сайт, але для тесту залишаємо "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ключ від Hugging Face (в реальності беремо з середовища os.getenv)
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL_MATCH = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return " ".join(text.split())

@app.post("/api/analyze")
async def analyze_resume(
    file: UploadFile = File(...), 
    vacancy_text: str = Form(...)
):
    try:
        # 1. Читаємо PDF
        file_bytes = await file.read()
        resume_text = extract_text_from_pdf(file_bytes)
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Не вдалося прочитати текст з PDF")

        # 2. Відправляємо на ШІ для порівняння (Match)
        payload = {"inputs": {"source_sentence": vacancy_text, "sentences": [resume_text]}}
        response = requests.post(API_URL_MATCH, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Помилка сервера ШІ")
            
        scores = response.json()
        match_percentage = round(scores[0] * 100, 1)

        # 3. Повертаємо красиву відповідь на фронтенд
        return {
            "status": "success",
            "match_score": match_percentage,
            "resume_preview": resume_text[:200] + "..." # Віддаємо шматочок тексту для прев'ю
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
