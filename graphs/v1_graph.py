from typing import Any

from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from models.schemas import ModuleQuery
from utils.llm import llm, vector_store

# # ---------- State Schema ----------
# class ModuleQuery(TypedDict):
#     moduleCodes: Annotated[List[str], ..., "Module codes mentioned in the query, if any."]

class State(TypedDict):
    question: str
    query: ModuleQuery
    context: list[Any]
    answer: str

prompt_template = hub.pull("rlm/rag-prompt")

def analyze_query(state: State):
    structured_llm = llm.with_structured_output(ModuleQuery)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

def retrieve(state: State):
    query = state["query"]
    if query["moduleCodes"]:
        filter_dict = {"moduleCode": {"$in": query["moduleCodes"]}}
        retrieved_docs = vector_store.similarity_search(query="", k=5, filter=filter_dict)
    else:
        retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(
        f"Code: {doc.metadata.get('moduleCode')}\n"
        f"Title: {doc.metadata.get('title')}\n"
        f"Department: {doc.metadata.get('department')}\n"
        f"Faculty: {doc.metadata.get('faculty')}\n"
        f"Credits: {doc.metadata.get('moduleCredit')}\n"
        f"Description: {doc.page_content}"
        for doc in state["context"]
    )
    messages = prompt_template.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph_v1 = graph_builder.compile()
