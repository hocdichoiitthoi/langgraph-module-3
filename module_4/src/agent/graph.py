from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START,END
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate

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


sg = StateGraph(GraphState)
sg.add_node("generate_summary", generate_summary)
sg.set_entry_point("generate_summary")
sg.set_finish_point("generate_summary") 
sub_app1 = sg.compile()


# -----------------------------
# Sub-graph: Generate Title
# -----------------------------

def generate_title(state: GraphState) -> GraphState:
    prompt_title = "Bạn là một AI tạo tiêu đề ngắn gọn. Hãy viết tiêu đề cho bản tóm tắt sau:\n\n{summary}"

    resp = llm.invoke(HumanMessage(content=prompt_title) + state.get("summary",""))
    return {"title": resp.content}


tg = StateGraph(GraphState)
tg.add_node("generate_title", generate_title)
tg.set_entry_point("generate_title")
tg.set_finish_point("generate_title")
sub_app2 = tg.compile()

# -----------------------------
# Supervisor Agent
# -----------------------------
def supervisor(state: GraphState) -> dict:
    if not state.get("summary"):
        return {"next": "summary_conversation"}
    elif not state.get("title"):
        return {"next": "generate_title"}
    else:
        return {"next": "val"}

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

    # Add supervisor + val
builder.add_node("supervisor", supervisor)
builder.add_node("val", val)

    # Add sub-graph
builder.add_node("summary_conversation", sub_app1)
builder.add_node("generate_title", sub_app2)

    # Luồng chính
builder.add_edge(START, "supervisor")
builder.add_edge("val", "supervisor")
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
graph = builder.compile()