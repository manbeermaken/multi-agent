import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

load_dotenv()

LLM_MODEL = "google_genai:gemini-3.5-flash-lite"
EMBEDDING_MODEL = "text-embedding-004"
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")

llm = init_chat_model(model=LLM_MODEL)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = InMemoryVectorStore(embeddings)

knowledge = [
    "LangChain is a toolkit for building LLM applications.",
    "LangGraph is meant for agent orchestration and durable processes."
]

vector_store.add_documents(documents=[Document(page_content=text) for text in knowledge])