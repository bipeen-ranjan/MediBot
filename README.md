# 🧠 MediBot – AI-Powered Medical Chatbot with RAG using LangChain and Hugging Face

MediBot is an AI-powered chatbot built using **LangChain**, **Hugging Face**, **FAISS**, and **Streamlit**. It leverages **Retrieval-Augmented Generation (RAG)** to answer medical queries based **only on uploaded medical PDFs**, avoiding hallucinations and ensuring responses are grounded in real, verified content.

---

## 📌 Features

- 💬 Ask medical questions in natural language
- 📚 Answers are retrieved from your uploaded PDF documents
- 🔎 Uses **FAISS** for fast vector-based semantic search
- 🤖 Uses **Hugging Face Transformers** for LLM responses
- 📄 Preprocesses large PDFs into chunks for accurate retrieval
- 🧠 Embeds chunks using Sentence Transformers

---

## 🛠️ Tech Stack

| Tool/Library      | Purpose                             |
|-------------------|-------------------------------------|
| `LangChain`       | Building RAG pipelines              |
| `FAISS`           | Storing and querying vector DB      |
| `HuggingFace`     | Hosting LLMs and embedding models   |
| `Streamlit`       | Web UI for interaction              |
| `Python` + `venv` | Core development & environment mgmt |

---

## 📂 Project Structure

MediBot/
├── data/                                        # Folder containing input PDFs
├── vectorstore/                                 # FAISS vector database
│ └── db_faiss/                                  # Stored embeddings
├── create_memory_for_llm.py                     # Script to load PDFs & store vector DB
├── connect_memory_with_llm.py                   # Query LLM with vector DB (RAG)
├── .env                                         # Hugging Face API key
└── README.md                                     # This file







## ⚙️ Setup Instructions

1. Create and Activate a Virtual Environment

python -m venv venv
venv\Scripts\activate      # Windows


2. Install Dependencies
 
# pip install -r requirements.txt

3. Add Your Hugging Face Token
Create a .env file in the root directory and add:

# HF_TOKEN=your_huggingface_api_token_here

You can get a free token at: https://huggingface.co/settings/tokens

## 🧠 Step 1: Create the Vector Memory (FAISS Index)

Run this script to load your PDFs, chunk text, generate embeddings, and save to FAISS:

python create_memory_for_llm.py

it will-
✅ Loads PDFs from data/
✅ Splits text into chunks
✅ Embeds each chunk
✅ Saves embeddings into FAISS (vectorstore/db_faiss)

## 🔍 Step 2: Interact with the Chatbot (RAG + LLM)
Once the FAISS vector store is ready, run:

python connect_memory_with_llm.py

Ask medical questions and get LLM responses grounded in your PDFs.

🌐 Optional: Run the Web App
Launch the chatbot interface via Streamlit:

streamlit run app.py

Make sure app.py is configured to load the correct FAISS DB and embedding model.


🧪 Sample Query
Input:
# What are the symptoms of malaria?

Response:
# Malaria symptoms typically include high fever, chills, sweating, headache, nausea, and muscle pain. These symptoms often appear 10–15 days after the bite of an infected mosquito.

## 📦 requirements.txt
langchain>=0.2.0
langchain-community>=0.2.0
langchain-huggingface>=0.0.1
huggingface_hub>=0.21.4
faiss-cpu>=1.7.4
sentence-transformers>=2.6.1
python-dotenv>=1.0.1
streamlit>=1.35.0

# Install using:

pip install -r requirements.txt

## 📄 Dataset Notes
Source: The Gale Encyclopedia of Medicine

Place your PDF files in the data/ folder.

Ensure PDFs are well-formatted (not scanned images) for accurate text extraction and chunking.

## ⚠️ Important Notes

-FAISS DB is local and persistent
-Responses are limited to your PDF data — no external knowledge used
-Works offline after embedding stage (Hugging Face API still required during query)
-Use from a trusted environment — deserialization involves pickle




















































