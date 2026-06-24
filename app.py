import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Page settings
st.set_page_config(page_title="PDF Chatbot")

st.title("📚 PDF Chatbot")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Split PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    st.success("PDF loaded successfully!")

    # Question input
    question = st.text_input(
        "Ask a question about the PDF"
    )

    # Ask button
    if st.button("Ask Question"):

        if question.strip():

            try:

                with st.spinner("Thinking..."):

                    # Use first few chunks as context
                    context = "\n".join(chunks[:5])

                    model = genai.GenerativeModel(
                        "gemini-3.5-flash"
                    )

                    prompt = f"""
You are a PDF assistant.

Answer ONLY from the PDF content provided below.

If the answer is not present in the PDF, say:
"I could not find that information in the PDF."

PDF Content:
{context}

Question:
{question}
"""

                    response = model.generate_content(
                        prompt
                    )

                    st.subheader("Answer")

                    st.write(response.text)

            except Exception as e:

                st.error(f"Error: {e}")

        else:

            st.warning(
                "Please enter a question."
            )