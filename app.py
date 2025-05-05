import os
import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
import logging
import asyncio

# ----------------------- SETUP -----------------------

# Load environment variables from .env file
load_dotenv()

# Suppress warnings and unnecessary logging
warnings.filterwarnings("ignore")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("google").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Streamlit/async fix
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Disable Streamlit file watchdog warning
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNING"] = "true"

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Use Gemini 1.5 Flash or 2.0 Flash (change here if needed)
model = genai.GenerativeModel("models/gemini-1.5-flash")  # or "models/gemini-2.0-flash"

# ----------------------- FUNCTIONS -----------------------

def load_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    else:
        return "Unsupported file format"

def split_into_chunks(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)

def highlight_risks(text):
    risk_words = ["penalty", "termination", "confidentiality", "liability", "breach", "fine"]
    return [word for word in risk_words if word.lower() in text.lower()]

def ask_gemini(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer briefly and clearly:"
    response = model.generate_content([{"role": "user", "parts": [{"text": prompt}]}])
    return response.text.strip() if hasattr(response, "text") else "No response received."

# ----------------------- STREAMLIT UI -----------------------

st.set_page_config(page_title="Doc QA with Gemini", layout="centered")
st.title("📄✍️ Document Analyzer with Gemini Flash")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file🤖", type=["txt", "pdf"])

if uploaded_file:
    with st.spinner("Reading document..."):
        text = load_text(uploaded_file)

    st.success("Document loaded successfully!")

    with st.spinner("Splitting into chunks..."):
        chunks = split_into_chunks(text)
        db = create_vector_db(chunks)

    st.success("Chunks processed and vector DB created.")

    st.subheader("🔍 Ask a question about the document")
    question = st.text_input("Your Question")

    if question:
        with st.spinner("Searching best context..."):
            top_docs = db.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in top_docs])

        with st.spinner("Generating response with Gemini Flash..."):
            answer = ask_gemini(question, context)
            st.markdown(f"💡 *Answer:* {answer}")

    with st.expander("🚨 Risky Terms Detected"):
        risks = highlight_risks(text)
        st.write(", ".join(set(risks)) if risks else "No critical risk terms found.")

    with st.expander("📋 Preview Extracted Text"):
        st.text(text[:2000] + "..." if len(text) > 2000 else text)
