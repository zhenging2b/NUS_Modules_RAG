from langgraph.graph import MessagesState, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from utils.llm import llm, vector_store
from langchain_core.tools import tool
from models.schemas import ModuleQuery

# Define tools
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    # Step A: structured parsing of query
    """Retrieve module information by code if present, otherwise semantic search."""
    structured_llm = llm.with_structured_output(ModuleQuery)
    modquery = structured_llm.invoke(query)

    # Step B: conditional retrieval
    if modquery["moduleCodes"]:
        filter_dict = {"moduleCode": {"$in": modquery["moduleCodes"]}}
        retrieved_docs = vector_store.similarity_search(
            query="",  # exact match, so semantic query can be empty
            k=5,
            filter=filter_dict
        )
    else:
        retrieved_docs = vector_store.similarity_search(query, k=5)

    # Step C: format return
    serialized = "\n\n".join(
        (
            f"Code: {doc.metadata.get('moduleCode')}\n"
            f"Title: {doc.metadata.get('title')}\n"
            f"Department: {doc.metadata.get('department')}\n"
            f"Faculty: {doc.metadata.get('faculty')}\n"
            f"Credits: {doc.metadata.get('moduleCredit')}\n"
            f"Description: {doc.page_content}"
        )
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve]) #LLM can "see" the tool once I bind it
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Step 2: Execute the retrieval. LangGraph utility node, something like if tool_call == "retrieve": run retrieve
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
           or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Build graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("query_or_respond", query_or_respond)
graph_builder.add_node("tools", tools)
graph_builder.add_node("generate", generate)
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges("query_or_respond", tools_condition, { "tools": "tools", "__end__": "__end__" })
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", "__end__")

checkpointer = MemorySaver()
graph_v2 = graph_builder.compile(checkpointer=checkpointer)
