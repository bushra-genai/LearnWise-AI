import os
import io
import time
import textwrap
from typing import List, Tuple

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Optional lightweight RAG (TF‑IDF) — keeps deps simple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(
    page_title="📚 Study Buddy (Groq + Open Source Models)",
    page_icon="📚",
    layout="wide",
)

# ---------------------------
# Helpers
# ---------------------------

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks for simple retrieval."""
    text = text.replace("\n", " ")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks


def extract_text_from_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(out)


def build_tfidf_index(texts: List[str]):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve_top_k(query: str, texts: List[str], vectorizer, matrix, k: int = 4) -> List[Tuple[int, float]]:
    if not texts:
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    idx_scores = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)
    return idx_scores[:k]


def format_context(chunks: List[str]) -> str:
    wrapped = [textwrap.fill(c, width=100) for c in chunks]
    return "\n\n".join([f"[Source #{i+1}]\n{c}" for i, c in enumerate(wrapped)])


def call_llm(model_name: str, api_key: str, system_prompt: str, messages: List[dict], temperature: float = 0.2, max_tokens: int = 1024) -> str:
    llm = ChatGroq(model=model_name, groq_api_key=api_key, temperature=temperature, max_tokens=max_tokens)

    lc_messages = [SystemMessage(content=system_prompt)]
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            # LangChain will treat prior assistant messages implicitly; we can skip or include as system-style context
            lc_messages.append(SystemMessage(content=f"Assistant said earlier: {m['content']}"))

    response = llm.invoke(lc_messages)
    return response.content if hasattr(response, "content") else str(response)


# ---------------------------
# Sidebar — Controls
# ---------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_key = st.text_input("🔑 Enter your GROQ API Key", type="password")
    st.caption("Get one from console.groq.com — we only use it locally.")

    model = st.selectbox(
        "Select Open‑Source Model (Groq)",
        [
            "gemma2-9b-it",
            "llama-3.1-8b-instant",
        ],
        index=0,
    )

    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)

    st.markdown("---")
    st.markdown("### 📄 Knowledge Sources (optional)")
    uploaded_files = st.file_uploader(
        "Upload PDFs / TXT notes (optional)", type=["pdf", "txt"], accept_multiple_files=True
    )

    enable_rag = st.checkbox("Use uploaded notes for context (RAG)", value=True)

    st.markdown("---")
    st.markdown("### 🎯 Study Goals")
    default_goal = "Prepare for midterm: Data Structures (Stacks, Queues, Trees), focus on time complexity and practice MCQs."
    study_goal = st.text_area("Your current study goal", value=default_goal)

    st.markdown("---")
    st.caption("Developed by Bushra")

# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # {role: user/assistant, content: str}

if "rag_texts" not in st.session_state:
    st.session_state.rag_texts = []
    st.session_state.vectorizer = None
    st.session_state.matrix = None

# Build / refresh RAG index when files change
if uploaded_files is not None and len(uploaded_files) > 0:
    collected_texts = []
    for f in uploaded_files:
        try:
            if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
                collected_texts.append(extract_text_from_pdf(f))
            else:
                collected_texts.append(f.read().decode("utf-8", errors="ignore"))
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")
    # Chunk
    chunks = []
    for t in collected_texts:
        chunks.extend(chunk_text(t))
    st.session_state.rag_texts = chunks

    if chunks:
        vectorizer, matrix = build_tfidf_index(chunks)
        st.session_state.vectorizer = vectorizer
        st.session_state.matrix = matrix

# ---------------------------
# Header
# ---------------------------
st.title("📚 LearnWise — AI Learning Partner") 
st.markdown(
    "Ask questions, generate flashcards & quizzes, and plan your study schedule — powered by open‑source LLMs via Groq."
)

# Quick actions bar
# Quick actions bar (3 columns now)
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🧭 Make Study Plan"):
        prompt = f"Create a concise, week-by-week study plan based on this goal: {study_goal}. Include topics, daily time blocks, and checkpoints."
        if not api_key:
            st.error("Please enter your GROQ API key in the sidebar.")
        else:
            out = call_llm(
                model, api_key,
                system_prompt=(
                    "You are Study Buddy: structured, encouraging, and practical. Keep outputs actionable."
                ),
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            st.session_state.messages.append({"role": "assistant", "content": out})
            st.success("Study plan generated below in the chat.")


with col2:
    if st.button("🧠 Generate Flashcards"):
        if not api_key:
            st.error("Please enter your GROQ API key in the sidebar.")
        elif not st.session_state.rag_texts:
            st.warning("Upload some notes or PDFs first.")
        else:
            sample = "\n\n".join(st.session_state.rag_texts[:20])
            user_prompt = f"""
            From the following notes, generate 15 flashcards in this exact format only:

            Q: [Question]
            A: [Answer]

            Keep answers under 30 words. No extra symbols, markdown, or headings.

            NOTES:
            {sample}
            """
            out = call_llm(
                model, api_key,
                system_prompt="You generate clear, exam-focused flashcards.",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.2,
            )
            st.session_state.messages.append({"role": "assistant", "content": out})
            st.success("Flashcards added to chat.")


with col3:
    if st.button("🧪 Create Quiz"):
        if not api_key:
            st.error("Please enter your GROQ API key in the sidebar.")
        elif not st.session_state.rag_texts:
            st.warning("Upload some notes or PDFs first.")
        else:
            sample = "\n\n".join(st.session_state.rag_texts[:20])
            user_prompt = f"""
            From the following notes, create a 10-question multiple-choice quiz. 
            Format it exactly like this:

            Q1. [Question text]
            A. Option 1
            B. Option 2
            C. Option 3
            D. Option 4
            Answer: [Correct option letter]

            Q2. ...

            NOTES:
            {sample}
            """
            out = call_llm(
                model, api_key,
                system_prompt="You are a strict examiner. Make challenging, clear MCQs with exactly one correct answer.",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.2,
            )
            st.session_state.messages.append({"role": "assistant", "content": out})
            st.success("Quiz added to chat.")


st.markdown("---")

# ---------------------------
# Chat Interface (with optional RAG)
# ---------------------------
st.subheader("💬  Chat With LearnWise")

user_input = st.chat_input("Ask anything about your course, notes, or exams…")

# Print history
for m in st.session_state.messages:
    with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
        st.markdown(m["content"]) 

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not api_key:
        with st.chat_message("assistant"):
            st.error("Please paste your GROQ API key in the sidebar to chat.")
    else:
        # Retrieve context if enabled
        context_snippets = []
        if enable_rag and st.session_state.rag_texts and st.session_state.vectorizer is not None:
            top = retrieve_top_k(user_input, st.session_state.rag_texts, st.session_state.vectorizer, st.session_state.matrix, k=4)
            context_snippets = [st.session_state.rag_texts[i] for i, _ in top]

        system_prompt = (
            "You are Study Buddy: a friendly, accurate tutor.\n"
            "Explain step-by-step, show formulas in LaTeX when needed, and prefer short paragraphs.\n"
            "If using context, cite [Source #] like [1], [2] based on provided snippets.\n"
        )

        # Compose user prompt with context
        if context_snippets:
            context_block = format_context(context_snippets)
            composed = (
                f"Use this context to answer. If irrelevant, answer from general knowledge and say 'Based on general knowledge'.\n\n"
                f"CONTEXT:\n{context_block}\n\nQUESTION: {user_input}"
            )
        else:
            composed = user_input

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = call_llm(
                    model_name=model,
                    api_key=api_key,
                    system_prompt=system_prompt,
                    messages=st.session_state.messages + [{"role": "user", "content": composed}],
                    temperature=temperature,
                )
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown("---")

# ---------------------------
# Export & Utilities
# ---------------------------
colx, coly = st.columns([3,1])
with colx:
    if st.session_state.messages:
        transcript = "\n\n".join([f"{m['role'].upper()}:\n{m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="💾 Download Chat (.txt)",
            data=transcript.encode("utf-8"),
            file_name="study_buddy_chat.txt",
            mime="text/plain",
        )
with coly:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------
# Footer Tip
# ---------------------------
st.caption(
    "Tip: Upload your lecture PDFs or handwritten notes (as text) and enable RAG to get grounded answers, flashcards, and quizzes."
)
