import streamlit as st
import os
import json
import re
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Personalized Doubt Solver",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Personalized Doubt Solver")
st.caption("AI Tutor + Smart Quiz + Dictionary + Weather + arXiv")

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in .env file.")
    st.stop()

# =========================================================
# OPENROUTER CLIENT
# =========================================================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL_NAME = "openrouter/free"

# =========================================================
# FREE APIs
# =========================================================

# Dictionary API
def get_word_meaning(word):
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url)
        if response.status_code != 200:
            return "Word not found."
        data = response.json()
        return data[0]["meanings"][0]["definitions"][0]["definition"]
    except:
        return "Meaning not found."

# Weather API (Open-Meteo)
def get_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=23.25&longitude=87.85&current_weather=true"
        data = requests.get(url).json()
        temp = data["current_weather"]["temperature"]
        wind = data["current_weather"]["windspeed"]
        return f"🌡 {temp}°C | 💨 {wind} km/h"
    except:
        return "Weather unavailable."

# arXiv API
def search_arxiv(query):
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3"
        response = requests.get(url)
        root = ET.fromstring(response.content)

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        results = []
        for entry in entries:
            title = entry.find("atom:title", ns).text.strip()
            link = entry.find("atom:id", ns).text.strip()
            results.append((title, link))
        return results
    except:
        return []

# =========================================================
# SESSION STATE
# =========================================================
def init_session():
    defaults = {
        "text_content": "",
        "quiz_questions": [],
        "current_q": 0,
        "answers": {},
        "quiz_score": None,
        "chat_history": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# =========================================================
# HELPERS
# =========================================================
def extract_text(file):
    content = ""
    if file.type == "text/plain":
        content = file.read().decode("utf-8")
    elif file.type == "application/pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            content += page.extract_text() or ""
    elif file.type in ["text/html", "application/octet-stream"]:
        soup = BeautifulSoup(file.read(), "html.parser")
        content = soup.get_text(separator="\n", strip=True)
    return content[:20000]


def generate_explanation(question, difficulty):
    st.session_state.chat_history.append({"role": "user", "content": question})

    messages = [
        {"role": "system", "content": f"You are a helpful tutor. Explain at {difficulty} level."}
    ] + st.session_state.chat_history

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.5
    )

    answer = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    return answer


# 🔥 FULLY ROBUST QUIZ GENERATOR
def generate_quiz(max_questions, difficulty):
    quiz_prompt = f"""
    Based strictly on this content:
    {st.session_state.text_content}

    Generate exactly {max_questions} MCQs at {difficulty} level.

    Return ONLY valid JSON array.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Output only valid JSON arrays. Ensure keys 'A', 'B', 'C', 'D' are used for options."},
            {"role": "user", "content": quiz_prompt}
        ],
        temperature=0.2
    )

    raw = response.choices[0].message.content.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)

    if not match:
        raise ValueError("Invalid quiz format returned.")

    quiz_data = json.loads(match.group(0))
    normalized = []

    for q in quiz_data:

        question = q.get("question", "No question provided.")

        # Normalize options
        options = {}

        if isinstance(q.get("options"), dict):
            options = q["options"]
        elif isinstance(q.get("options"), list):
            options = {chr(65+i): opt for i, opt in enumerate(q["options"])}
        elif all(k in q for k in ["A", "B", "C", "D"]):
            options = {
                "A": q["A"],
                "B": q["B"],
                "C": q["C"],
                "D": q["D"]
            }
        elif isinstance(q.get("choices"), list):
            options = {chr(65+i): opt for i, opt in enumerate(q["choices"])}
        else:
            options = {
                "A": "Option not available",
                "B": "Option not available",
                "C": "Option not available",
                "D": "Option not available"
            }

        # ROBUST ANSWER MATCHING FIX
        raw_correct = str(q.get("correct") or q.get("answer") or "A").strip()
        correct_key = "A" # Default

        # 1. Check if the AI returned standard keys like "A", "b", "C"
        if raw_correct.upper() in options.keys():
            correct_key = raw_correct.upper()
        # 2. Check if the AI returned the full option text (e.g. "Paris" instead of "A")
        else:
            for key, val in options.items():
                if raw_correct.lower() in str(val).lower():
                    correct_key = key
                    break
            # 3. Check if the AI returned "A. Option Text"
            if correct_key == "A" and len(raw_correct) >= 1:
                if raw_correct[0].upper() in options.keys():
                    correct_key = raw_correct[0].upper()

        normalized.append({
            "question": question,
            "options": options,
            "correct": correct_key
        })

    return normalized

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")
    difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])
    max_questions = st.slider("Quiz Questions", 3, 10, 5)

    st.divider()
    st.subheader("🌤 Live Weather")
    if st.button("Check Weather"):
        st.info(get_weather())

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["📖 Upload", "❓ Doubt", "🧠 Quiz"])

# =========================================================
# TAB 1 — Upload
# =========================================================
with tab1:
    file = st.file_uploader("Upload TXT, PDF or HTML", type=["txt", "pdf", "html"])
    if file:
        st.session_state.text_content = extract_text(file)
        st.success("Content loaded successfully!")
        
        # SNIPPET FEATURE ADDED HERE
        with st.expander("📄 Preview Extracted Text Snippet", expanded=True):
            st.write(st.session_state.text_content[:1500] + "... [Text truncated for preview]")

# =========================================================
# TAB 2 — Doubt + Dictionary + arXiv
# =========================================================
with tab2:
    if not st.session_state.text_content:
        st.info("Upload content first.")
    else:
        question = st.text_area("Enter your doubt")
        if st.button("Explain"):
            st.markdown(generate_explanation(question, difficulty))

        st.divider()
        st.subheader("📖 Dictionary")
        word = st.text_input("Enter word")
        if st.button("Get Meaning"):
            st.info(get_word_meaning(word))

        st.divider()
        st.subheader("🔬 arXiv Research Search")
        query = st.text_input("Search research papers")
        if st.button("Search arXiv"):
            papers = search_arxiv(query)
            if papers:
                for title, link in papers:
                    st.markdown(f"- [{title}]({link})")
            else:
                st.warning("No papers found.")

# =========================================================
# TAB 3 — FULLY REWRITTEN QUIZ ENGINE
# =========================================================
with tab3:

    st.subheader("🧠 Smart Quiz")

    if not st.session_state.text_content:
        st.info("Upload content first.")
        st.stop()

    # -------------------------
    # Generate Quiz
    # -------------------------
    if st.button("Generate New Quiz"):
        try:
            quiz = generate_quiz(max_questions, difficulty)

            # Reset state safely
            st.session_state.quiz_questions = quiz
            st.session_state.answers = {}
            st.session_state.quiz_score = None

            st.success("Quiz generated successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Quiz generation failed: {e}")
            st.stop()

    # -------------------------
    # If Quiz Exists
    # -------------------------
    if st.session_state.quiz_questions:

        quiz_data = st.session_state.quiz_questions

        st.markdown("### Answer All Questions")

        # Render all questions at once (better UX)
        for idx, q in enumerate(quiz_data):

            st.markdown(f"#### Question {idx+1}")
            st.write(q["question"])

            options = list(q["options"].keys())

            selected = st.radio(
                "Select answer:",
                options=options,
                format_func=lambda x: f"{x}: {q['options'][x]}",
                key=f"question_{idx}"
            )

            st.session_state.answers[idx] = selected
            st.divider()

        # -------------------------
        # Submit Quiz
        # -------------------------
        if st.button("Submit Quiz"):

            score = 0

            for idx, q in enumerate(quiz_data):

                user_answer = str(st.session_state.answers.get(idx, "")).strip().upper()
                correct_answer = str(q["correct"]).strip().upper()

                # Final normalization safety
                match = re.search(r"[A-D]", correct_answer)
                if match:
                    correct_answer = match.group(0)

                if user_answer == correct_answer:
                    score += 1

            st.session_state.quiz_score = score
            st.rerun()

        # -------------------------
        # Results Section
        # -------------------------
        if st.session_state.quiz_score is not None:

            total = len(quiz_data)
            score = st.session_state.quiz_score

            percentage = (score / total) * 100

            st.success(f"🎯 Final Score: {score} / {total}")
            st.progress(score / total)

            if percentage >= 80:
                st.info("🏆 Excellent Performance")
            elif percentage >= 50:
                st.info("👍 Good Performance")
            else:
                st.info("📘 Needs Improvement")

            # Detailed Breakdown
            st.markdown("### 📊 Answer Breakdown")

            result_data = []

            for idx, q in enumerate(quiz_data):

                user_answer = st.session_state.answers.get(idx)
                correct_answer = q["correct"]

                is_correct = str(user_answer).strip().upper() == str(correct_answer).strip().upper()

                result_data.append({
                    "Question": f"Q{idx+1}",
                    "Your Answer": user_answer,
                    "Correct Answer": correct_answer,
                    "Result": "Correct" if is_correct else "Wrong"
                })

            df = pd.DataFrame(result_data)
            st.dataframe(df, use_container_width=True)

            # Restart
            if st.button("Start New Quiz"):
                st.session_state.quiz_questions = []
                st.session_state.answers = {}
                st.session_state.quiz_score = None
                st.rerun()