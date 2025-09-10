# bài_1_streaming_llm.py

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
import asyncio
# -------------------------------
# 1. Định nghĩa state
# -------------------------------
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# -------------------------------
# 2. Khởi tạo LLM (có stream)
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------------
# 3. Node sinh phản hồi từ LLM
# -------------------------------
def call_llm(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# -------------------------------
# 4. Xây dựng graph
# -------------------------------
workflow = StateGraph(State)
workflow.add_node("chatbot", call_llm)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

app = workflow.compile()

# -------------------------------
# 5. Thực thi với astream()
# -------------------------------
async def main():
    print(">>> Streaming response...\n")

    input_state = {"messages": [HumanMessage(content="Viết một đoạn giới thiệu ngắn về AI.")]}

    async for event in app.astream_events(input_state, version="v2"):
    # Get chat model tokens from a particular node 
       if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == "chatbot":
           data=event["data"]
           print(data["chunk"].content, end="", flush=True)
    print("\n\n>>> Done.")

if __name__ == "__main__":
    asyncio.run(main())