from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from utils.config import GOOGLE_API_KEY, LANGSMITH_API_KEY

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="./nus_modules_index",
)
