# %%
# Environment Setup
import os
from dotenv import load_dotenv
# Load Multiple PDFs
from langchain.document_loaders import PyPDFLoader
# Split Documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Vector Store
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Chains
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# %%
load_dotenv()

# %%
google_api_key = os.getenv("GOOGLE_API_KEY")

# %%
pdf_folder = "RAGData"
documents = []

# %%
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        documents.extend(loader.load()) 
        print(f"Loaded {len(documents)} documents from {file}")

# %%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

# %%
docs = text_splitter.split_documents(documents)

# %%
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# %%
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./chroma_db"  # Persistent local dir
)

# %%
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=google_api_key)

# %%
retriever = vectorstore.as_retriever()

# %%
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# %%
query = "What Foreign Jurisdiction means?"

# %%
response = qa_chain.invoke(query)

# %%
import google.generativeai as genai

genai.configure(api_key=google_api_key)
models = genai.list_models()
for model in models:
    print(model.name)

# %%



