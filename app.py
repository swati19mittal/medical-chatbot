import os
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import warnings
import logging

# Suppress warnings and fix tokenizer parallelism
warnings.filterwarnings('ignore')
logging.getLogger('llama_cpp').setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['GGML_METAL_LOG_LEVEL'] = '0'

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Updated LLM with safer parameters
llm = LlamaCpp(
    model_path="model/llama-2-7b-chat.Q4_0.gguf",
    n_ctx=512,           # Reduced context window
    n_batch=8,           # Much smaller batch size
    temperature=0.8,
    max_tokens=256,      # Shorter responses
    verbose=False,
    n_threads=1,         # Single thread to avoid race conditions
    n_gpu_layers=0       # Force CPU only
)

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_query = msg
    print(input_query)
    result = qa.invoke({"query": input_query})
    print("Response:", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)


