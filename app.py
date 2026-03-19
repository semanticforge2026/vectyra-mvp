import streamlit as st
import requests
import fitz  # PyMuPDF
from supabase import create_client, Client

# ==========================================
# НАЛАШТУВАННЯ КЛЮЧІВ ТА API (БЕЗПЕЧНИЙ РЕЖИМ)
# ==========================================

# 1. Ключі від DBA (Supabase) беремо з секретів Streamlit
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Ключ від ШІ (Hugging Face) також беремо з секретів
API_TOKEN = st.secrets["HUGGING_FACE_KEY"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# URL для генерації векторів (збереження в базу)
API_URL_EMBEDDING = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
# URL для порівняння текстів (Semantic Match)
API_URL_MATCH = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# БАЗОВІ ФУНКЦІЇ (ЛОГІКА)
# ==========================================

def extract_text_from_bytes(file_bytes):
    """Читає PDF прямо з оперативної пам'яті (для Streamlit)"""
    text = ""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text") + "\n"
        return " ".join(text.split())
    except Exception as e:
        st.error(f"Помилка читання PDF: {e}")
        return None


def get_embedding(text):
    """Генерує ШІ-вектор з тексту для бази даних"""
    payload = {"inputs": text}
    response = requests.post(API_URL_EMBEDDING, headers=headers, json=payload)

    if response.status_code != 200:
        st.error(f"Помилка ШІ (Вектор): {response.text}")
        return None

    data = response.json()
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], list):
            return data[0]
        return data
    return None


def get_match_scores(source_text, target_texts):
    """Вираховує відсоток збігу між вакансією та резюме"""
    payload = {"inputs": {"source_sentence": source_text, "sentences": target_texts}}
    response = requests.post(API_URL_MATCH, headers=headers, json=payload)

    if response.status_code != 200:
        st.error(f"Помилка ШІ (Match): {response.text}")
        return None
    return response.json()


# ==========================================
# ВІЗУАЛЬНА ЧАСТИНА (STREAMLIT)
# ==========================================

st.set_page_config(page_title="Vectyra AI", page_icon="🚀", layout="centered")
st.title("🚀 Vectyra AI: Рекрутинг майбутнього")

# Створюємо дві вкладки для зручності
tab1, tab2 = st.tabs(["📂 Додавання кандидата в БД", "🎯 Аналіз збігу (Match)"])

# --- ВКЛАДКА 1: ПАРСЕР PDF ТА БАЗА ДАНИХ ---
with tab1:
    st.subheader("Завантаження PDF-резюме")
    st.write("ШІ автоматично прочитає файл, згенерує вектор і збереже в Supabase.")

    uploaded_file = st.file_uploader("Оберіть файл резюме (.pdf)", type=["pdf"])

    if uploaded_file is not None:
        if st.button("💾 Обробити та зберегти", type="primary", key="save_btn"):
            with st.spinner("⏳ Читаємо PDF-документ..."):
                file_bytes = uploaded_file.read()
                resume_text = extract_text_from_bytes(file_bytes)

            if resume_text:
                st.success("✅ Текст витягнуто!")

                with st.spinner("🧠 ШІ генерує вектор..."):
                    embedding_vector = get_embedding(resume_text)

                if embedding_vector:
                    with st.spinner("💾 Записуємо в Supabase..."):
                        try:
                            supabase.table('resumes').insert({
                                "content": resume_text,
                                "embedding": embedding_vector
                            }).execute()
                            st.balloons()
                            st.success("🎉 Резюме успішно додано до бази даних Vectyra!")
                        except Exception as e:
                            st.error(f"❌ Помилка запису в БД: {e}")

# --- ВКЛАДКА 2: СЕМАНТИЧНИЙ ЗБІГ ---
with tab2:
    st.subheader("Семантичне порівняння")
    st.markdown("Введіть опис вакансії та текст резюме кандидата для швидкої перевірки.")

    vacancy_text = st.text_area("📝 Опис вакансії:", height=150, placeholder="Введіть вимоги до кандидата...")
    candidate_text = st.text_area("👤 Досвід кандидата (Резюме):", height=150,
                                  placeholder="Введіть досвід роботи та навички...")

    if st.button("🚀 Аналізувати збіг", type="primary", key="match_btn"):
        if vacancy_text and candidate_text:
            with st.spinner("ШІ аналізує семантику..."):
                scores = get_match_scores(vacancy_text, [candidate_text])

                if scores:
                    match_percentage = round(scores[0] * 100, 1)
                    st.markdown("---")
                    st.subheader("Результат аналізу:")

                    if match_percentage > 60:
                        st.success(f"✅ Match Score: {match_percentage}% (Високий збіг)")
                    elif match_percentage > 40:
                        st.warning(f"⚠️ Match Score: {match_percentage}% (Частковий збіг)")
                    else:
                        st.error(f"❌ Match Score: {match_percentage}% (Низький збіг)")
        else:
            st.warning("Будь ласка, заповніть обидва поля!")
