import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import random
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
PDF_PATH = "/Users/parimalkashireddy/Desktop/rag_med/Data/The_Gale_Encyclopedia_of_Medicine.pdf"
DB_FAISS_PATH = "vectorstore/db_faiss"
# CUSTOM_PROMPT_TEMPLATE = """
# Use the context to provide clear and medically grounded answers.
# If unsure, say "I don't know" instead of guessing.

# Context: {context}
# Question: {question}

# Answer:
# """
CUSTOM_PROMPT_TEMPLATE = """
You are a medical expert assistant.  
Given the following context, explain the answer in 3–5 complete sentences.  
Avoid vague or one-word answers.  
If you do not know the answer based on the context, say "I don't know" politely.

Context: {context}
Question: {question}

Answer:
"""
@st.cache_resource(show_spinner=False)
def load_random_questions(filepath="questions.txt", n=5):
    """Load questions from a file and pick n random questions."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    questions = [line.strip().strip('"') for line in lines if line.strip()]
    return random.sample(questions, n)
@st.cache_resource
def load_or_build_vectorstore():
    if os.path.exists(DB_FAISS_PATH):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(DB_FAISS_PATH)

    return vectorstore

@st.cache_resource
def get_local_llm():
    model_name = "tiiuae/falcon-rw-1b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    local_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=512,
        temperature=0.7
        # max_length=512

    )

    return local_pipe

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def main():
    st.set_page_config(page_title="🩺 Adaptive RAG Medical Chatbot", layout="wide")
    st.title("Adaptive RAG Medical Chatbot (Local + Cached)")
    st.write("Ask questions based on **The Gale Encyclopedia of Medicine**.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "autofill_prompt" not in st.session_state:
        st.session_state.autofill_prompt = ""
    st.subheader("Suggested Questions:")
    try:
        suggestions = load_random_questions(filepath="questions.txt", n=5)
        for q in suggestions:
            if st.button(q, key=q):
                st.session_state.autofill_prompt = q
    except Exception as e:
        st.error(f"Failed to load questions: {e}")
    prompt = st.text_input("What do you want to know?", key="chat_input", value=st.session_state.autofill_prompt)

    if not prompt:
        return
    st.session_state.autofill_prompt = ""
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("🔎 Searching medical knowledge..."):
        vectorstore = load_or_build_vectorstore()
        retriever   = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Retrieve top-3 chunks
        docs = retriever.invoke(prompt)
        context = "\n\n".join([d.page_content for d in docs])
        llm = get_local_llm()
        full_prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context, question=prompt)

        output = llm(full_prompt)
        result = output[0]["generated_text"].strip()
    st.chat_message("assistant").markdown(result)
    st.session_state.messages.append({"role": "assistant", "content": result})
    with st.expander("See Retrieved Contexts"):
        for i, d in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:** {d.page_content}")

if __name__ == "__main__":
    main()
