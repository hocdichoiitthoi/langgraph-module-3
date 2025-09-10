import asyncio
from typing import Annotated, TypedDict
import PyPDF2
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# ===============================
# 1. Khai báo State
# ===============================
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    approved: bool

# ===============================
# 2. Khai báo model
# ===============================
llm = AzureChatOpenAI(
    model="gpt-4.1",
    api_version="2025-01-01-preview", 
    temperature=0)

# ===============================
# 3. Các node chính
# ===============================
# Node AI: viết email trả lời
def summarize_doc(state: State):
    text = state["messages"][-1].content
    feedback = ""
    if len(state["messages"]) > 1:
        feedbacks = [m.content for m in state["messages"][1:]]
        feedback = "\n".join(feedbacks)

    # Prompt gốc + feedback
    prompt = f"Tóm tắt văn bản sao cho vẫn nắm được ý chính:\n\n{text}\n\n"
    if feedback:
        prompt += f"Yêu cầu chỉnh sửa bổ sung: {feedback}"

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

def human_feedback(state: State, approved: bool, feedback_text: str = ""):
    if approved:
        return {"approved": True}
    else:
        return {"messages": state["messages"] + [HumanMessage(content=f"Refine lại tóm tắt: {feedback_text}")], "approved": False}

# Node Save: lưu tóm tắt
def save_summary(state: State):
    print("\n✅ Tóm tắt cuối cùng được lưu!")
    print(state["messages"][-1].content)
    return {}

def decide_next(state: State):
    if state["approved"] == True:
        return "save_summary"
    else:
        return "feedback"
# ----------------------------
# 3. Xây workflow
# ----------------------------
graph = StateGraph(State)
graph.add_node("summarize", summarize_doc)
graph.add_node("feedback", human_feedback)
graph.add_node("save", save_summary)

graph.add_edge(START, "summarize")
graph.add_conditional_edges(
    "summarize", 
    decide_next, 
    {"save_summary": "save", "human_feedback": "feedback"})
graph.add_edge("feedback","summarize")    
graph.add_edge("save", END)

memory = MemorySaver()
app_graph = graph.compile(checkpointer=memory)
