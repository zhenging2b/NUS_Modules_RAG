from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain.chat_models import init_chat_model
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env file into environment variables

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------- Setup ----------
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="./nus_modules_index",
)

prompt_template = hub.pull("rlm/rag-prompt")


# ---------- State Schema ----------
class ModuleQuery(TypedDict):
    moduleCodes: Annotated[List[str], ..., "Module codes mentioned in the query, if any."]


class State(TypedDict):
    question: str
    query: ModuleQuery
    context: List[Any]
    answer: str


# ---------- Workflow ----------
def analyze_query(state: State):
    structured_llm = llm.with_structured_output(ModuleQuery)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    query = state["query"]

    if query["moduleCodes"]:
        filter_dict = {"modeuleCode": {"$in": query["moduleCodes"]}}
        retrieved_docs = vector_store.similarity_search(
            query="", k=5, filter=filter_dict
        )
    else:
        retrieved_docs = vector_store.similarity_search(state["question"])

    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(
        f"Code: {doc.metadata.get('modeuleCode')}\n"
        f"Title: {doc.metadata.get('title')}\n"
        f"Department: {doc.metadata.get('department')}\n"
        f"Faculty: {doc.metadata.get('faculty')}\n"
        f"Credits: {doc.metadata.get('moduleCredit')}\n"
        f"Description: {doc.page_content}"
        for doc in state["context"]
    )
    messages = prompt_template.invoke(
        {"question": state["question"], "context": docs_content}
    )
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()


# ---------- FastAPI App ----------
app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(req: QuestionRequest):
    result = None
    for step in graph.stream({"question": req.question}, stream_mode="updates"):
        result = step  # keep the last update
    return {"result": result}
@app.get("/")
def home():
    return {"message": "FastAPI server is running. Use POST /ask to query."}