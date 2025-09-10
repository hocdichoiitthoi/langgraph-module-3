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

# Node Human: duyệt hoặc chỉnh sửa
def human_feedback(state: State):
    choice = input("\nBạn có đồng ý với tóm tắt này không? (y/n): ")
    '''while choice != "y" or choice != "n":
        choice = input("Không hợp lệ, vui lòng nhập lại! (y/n): ")'''
    if choice.lower() == "y":
        return {"approved": True}
    else:
        feedback = input("Hãy nhập phản hồi / chỉnh sửa mong muốn: ")
        return {"messages": state["messages"] + [HumanMessage(content=f"Refine lại tóm tắt: {feedback}")], "approved": False}

# Node Save: lưu tóm tắt
def save_summary(state: State):
    print("\n✅ Tóm tắt cuối cùng được lưu!")
    print(state["messages"][-1].content)
    return {}

def decide_next(state: State):
    if state["approved"] == True:
        return "save_summary"
    else:
        return "human_feedback"
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

# ----------------------------
# 4. Chạy thử
# ----------------------------

'''def read_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text'''

async def main():
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == "exit":
            break
        print("Agent: ", end="", flush=True)
        # document = read_pdf("module3/files/LVW.pdf")
        thread = {"configurable": {"thread_id": "1"}}
        state = {"messages": [HumanMessage(content=user_input)], "approved": False}
        async for event in app_graph.astream_events(state, thread, version="v2"):
        # Bắt sự kiện streaming từ LLM
            if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == "summarize":
                data=event["data"]
                print(data["chunk"].content, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
