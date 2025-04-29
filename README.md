# Medical_RAG_Chatbot
A lightweight Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, FAISS, Hugging Face Transformers, and LangChain.

## Environment Setup

Make sure you activate your environment before running any code:

```bash
conda env create -f rag.yml
conda activate llms
```
## Running using Huggingface token
```bash
export HF_TOKEN="hf_your_actual_token_here"
```
-> rag.py has the ability to access the saved variables using os.
-> If you want to try using someother medical related book just change the data path in code.
-> To try using a different huggingface model you can just change the model name when initiliazing it and in get_llm() method.
-> If you do not want to export HF_TOKEN in local you can modify the python line saying default initialization:
```bash 
HF_TOKEN = os.environ.get("HF_TOKEN","your_token_string_here")
```

## Running the model on local
```bash
streamlit run scripts/app.py --server.fileWatcherType none
```

# Try with your own dataset

The app splits your medical textbook into smaller chunks and builds a FAISS vectorstore.

The FAISS index and associated .pkl file are saved under the vectorstore/db_faiss/ folder.

These chunks are embedded into a format that Hugging Face models can easily retrieve and understand.

The LLM can either be loaded:

Locally (if you downloaded the model to your machine).

Or dynamically using the Hugging Face Hub with an API token.

Suggested Questions: Every time the app loads, a new random set of questions is displayed for easier exploration.

You can play with temperature and batch size, the higher temperature more creative model with perform and it may go out of context if temperature is very high.

