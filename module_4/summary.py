from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# -----------------------------
# Định nghĩa State
# -----------------------------
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    title: str
    feedback: str

# -----------------------------
# Tạo LLM
# -----------------------------
llm = AzureChatOpenAI(
    model="gpt-4.1", 
    api_version="2025-01-01-preview"
)

# -----------------------------
# Sub-graph: Summary Conversation
# -----------------------------
def generate_summary(state: GraphState) -> GraphState:
    prompt_summary = "Bạn là một trợ lý AI. Nhiệm vụ của bạn là tóm tắt hội thoại."

    resp = llm.invoke(HumanMessage(content=prompt_summary))
    return {"summary": resp.content}

def build_summary_graph():
    sg = StateGraph(GraphState)
    sg.add_node("generate_summary", generate_summary)
    sg.set_entry_point("generate_summary")
    sg.set_finish_point("generate_summary")
    return sg.compile()

# -----------------------------
# Sub-graph: Generate Title
# -----------------------------


def generate_title(state: GraphState) -> GraphState:
    prompt_title = "Bạn là một AI tạo tiêu đề ngắn gọn. Hãy viết tiêu đề cho bản tóm tắt sau:\n\n{summary}"

    resp = llm.invoke(HumanMessage(content=prompt_title) + state.get("summary",""))
    return {"title": resp.content}

def build_title_graph():
    tg = StateGraph(GraphState)
    tg.add_node("generate_title", generate_title)
    tg.set_entry_point("generate_title")
    tg.set_finish_point("generate_title")
    return tg.compile()

# -----------------------------
# Supervisor Agent
# -----------------------------
def supervisor(state: GraphState) -> str:
    if not state.get("summary"):
        return "summary_conversation"
    elif not state.get("title"):
        return "generate_title"
    else:
        return "val"

# -----------------------------
# Validation Agent (VAL)
# -----------------------------
def val(state: GraphState) -> GraphState:
    feedback = f"✅ Kết quả cuối:\n\nTóm tắt: {state.get('summary')}\n\nTiêu đề: {state.get('title')}"
    return {"feedback": feedback}

# -----------------------------
# Build Parent Graph
# -----------------------------
builder = StateGraph(GraphState)

    # Sub-graph
summary_graph = build_summary_graph()
title_graph = build_title_graph()

    # Add supervisor + val
builder.add_node("supervisor", supervisor)
builder.add_node("val", val)

    # Add sub-graph
builder.add_node("summary_conversation", summary_graph)
builder.add_node("generate_title", title_graph)

    # Luồng chính
builder.set_entry_point("supervisor")
builder.add_edge("val", END)

    # Supervisor điều hướng
builder.add_conditional_edges(
        "supervisor",
        supervisor,
        {
            "summary_conversation": "summary_conversation",
            "generate_title": "generate_title",
            "val": "val",
        },
    )

    # Sau khi chạy sub-graph -> quay lại supervisor
builder.add_edge("summary_conversation", "supervisor")
builder.add_edge("generate_title", "supervisor")
memory = MemorySaver()
app = builder.compile(checkpointer=memory)

# -----------------------------
# Demo chạy thử
# -----------------------------
if __name__ == "__main__":
    input_state = {"messages": ["Xin chào, tôi muốn đặt pizza và một chai Pepsi."]}
    
    # Run graph
    final_state = app.invoke(input_state)
    
    # In ra toàn bộ state để debug

    print(final_state)
    
