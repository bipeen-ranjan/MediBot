
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data):
    print("📂 Loading PDFs from:", data)
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages.")
    return documents

documents = load_pdf_files(DATA_PATH)

# Step 2: Create Chunks
def create_chunks(extracted_data):
    print("🔗 Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"✅ Created {len(text_chunks)} text chunks.")
    return text_chunks

text_chunks = create_chunks(documents)

# Step 3: Create Vector Embeddings
def get_embedding_model():
    print("📌 Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedding model loaded.")
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
print("💾 Saving to FAISS DB...")
DB_FAISS_PATH = "vectorstore/db_faiss"
os.makedirs(DB_FAISS_PATH, exist_ok=True)  # ensure folder exists
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print("✅ FAISS DB saved at:", DB_FAISS_PATH)
