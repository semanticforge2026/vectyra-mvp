import streamlit as st
import requests

# 1. Твій API ключ
API_TOKEN = "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"
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def get_match_scores(source_text, target_texts):
    payload = {"inputs": {"source_sentence": source_text, "sentences": target_texts}}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return None
    return response.json()


# ==========================================
# ВІЗУАЛЬНА ЧАСТИНА (STREAMLIT)
# ==========================================

# Налаштування сторінки
st.set_page_config(page_title="Vectyra AI", page_icon="🎯", layout="centered")

# Заголовок
st.title("🎯 Vectyra AI: Semantic Matching")
st.markdown("Введіть опис вакансії та текст резюме кандидата, щоб система вирахувала їх семантичний збіг.")

# Поля для вводу тексту
vacancy_text = st.text_area("📝 Опис вакансії:", height=150, placeholder="Введіть вимоги до кандидата...")
candidate_text = st.text_area("👤 Досвід кандидата (Резюме):", height=150,
                              placeholder="Введіть досвід роботи та навички...")

# Кнопка запуску
if st.button("🚀 Аналізувати збіг", type="primary"):
    if vacancy_text and candidate_text:
        # Спінер очікування
        with st.spinner("ШІ аналізує семантику..."):
            scores = get_match_scores(vacancy_text, [candidate_text])

            if scores:
                match_percentage = round(scores[0] * 100, 1)

                # Гарний вивід результату залежно від відсотка
                st.markdown("---")
                st.subheader("Результат аналізу:")

                if match_percentage > 60:
                    st.success(f"✅ Match Score: {match_percentage}% (Високий збіг)")
                elif match_percentage > 40:
                    st.warning(f"⚠️ Match Score: {match_percentage}% (Частковий збіг)")
                else:
                    st.error(f"❌ Match Score: {match_percentage}% (Низький збіг)")
            else:
                st.error("Виникла помилка при зверненні до ШІ. Перевірте API ключ.")
    else:
        st.warning("Будь ласка, заповніть обидва поля!")