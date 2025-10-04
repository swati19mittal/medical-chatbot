from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import warnings
import logging
import os

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('llama_cpp').setLevel(logging.ERROR)
os.environ['GGML_METAL_LOG_LEVEL'] = '0'


# Extract data from PDFs
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
