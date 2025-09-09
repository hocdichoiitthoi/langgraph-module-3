from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import List
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

llm = AzureChatOpenAI(
    model="gpt-4.1", 
    api_version="2025-01-01-preview",
    temperature=0
)

def chatbot_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)
memory=MemorySaver()
app = workflow.compile(checkpointer=memory)

async def main():
    print(">>> Chatbot mini \n")
    
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == "goodbye":
            print("AI: Bye, see you next time!", end="", flush=True)
            break

        config = {"configurable": {"thread_id": "1"}}
        input_state = {"messages": user_input}
        
        print("AI: ", end="", flush=True)
        async for event in app.astream_events(input_state, config, version="v2"):
           if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == "chatbot":
              data=event["data"]
              print(data["chunk"].content, end="", flush=True)
        print("")

if __name__ == "__main__":
    asyncio.run(main())              