import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# Configs
PDF_PATH = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# HUGGINFACE_REPO_ID = "tiiuae/falcon-rw-1b"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the context to provide clear and medically grounded answers.
If unsure, say "I don't know" instead of guessing.

Context: {context}
Question: {question}

Answer:
"""

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

def get_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation", 
        temperature=0.3,
        model_kwargs={
            "token": os.environ.get("HF_TOKEN"),
            "max_length": 512
        }
    )

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def main():
    st.title("🩺 Adaptive RAG Medical Chatbot")
    st.write("Ask questions based on The Gale Encyclopedia of Medicine.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    prompt = st.chat_input("What do you want to know?")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Searching medical knowledge..."):
            vectorstore = load_or_build_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            qa_chain = RetrievalQA.from_chain_type(
                llm=get_llm(),
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({"query": prompt})
            result = response["result"]

            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    main()
