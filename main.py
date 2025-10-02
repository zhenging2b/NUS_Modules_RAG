from fastapi import FastAPI
from pydantic import BaseModel
from graphs.v1_graph import graph_v1
from graphs.v2_graph import graph_v2
import uuid


# ---------- FastAPI App ----------
app = FastAPI()


class QuestionRequest(BaseModel):
    question: str
    thread_id: str | None = None


@app.post("/ask-v1")
def ask_question(req: QuestionRequest):
    result = None
    for step in graph_v1.stream({"question": req.question}, stream_mode="updates"):
        result = step  # keep the last update
    return {"result": result}


@app.post("/ask-v2")
def ask_v2(req: QuestionRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = None
    for step in graph_v2.stream({"messages": [{"role": "user", "content": req.question}]}, config=config, stream_mode="updates"):
        result = step
    return {"thread_id": thread_id, "result": result}

@app.get("/")
def home():
    return {"message": "FastAPI server is running. Use POST /ask to query."}