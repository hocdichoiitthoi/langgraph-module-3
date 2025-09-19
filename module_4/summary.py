from typing import List, TypedDict, Annotated, Literal
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
import os
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# ==============================
# 1. Khai báo state
# ==============================
class GraphState(TypedDict):
    messages: Annotated[AnyMessage, add_messages]
    summary: str
    title: str
    feedback_summary: str
    feedback_title: str

llm = AzureChatOpenAI(
    model="gpt-4.1",
    api_version="2025-01-01-preview", 
    temperature=0)

# ==============================
# 3. Summary Agent
# ==============================
# class SummaryState(TypedDict):
#   messages: Annotated[AnyMessage, add_messages]
#   summary: str
#   feedback_summary: str

# class SummaryOutputstate(TypedDict):
#   summary: str

#=== SUMMARY NODE ===

@tool
def generate_summary(state: GraphState):
    '''This node will summary a paragraph from the user input'''
    # Access the latest human message content as the text to summarize
    text_to_process = ""
    for message in state["messages"]:
       if isinstance(message, HumanMessage) and not message.content.lower().startswith(("feedback summary:", "feedback title:")):
        text_to_process = message.content
        break

    # Access the latest feedback on the summary if available
    feedback_summary = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage) and message.content.lower().startswith("feedback summary:"):
            feedback_summary = message.content[len("feedback summary:"):].strip()
            break


    prompt_summary = f"Summarize the following paragraph while keeping the main idea:\n{text_to_process}\n"

    if feedback_summary:
        prompt_summary += f"\nAdditional requirements for the summary: {feedback_summary}"

    response = llm.invoke(HumanMessage(content=prompt_summary))
    return {"summary": response.content}

#=== TITLE NODE ===

@tool
def generate_title(state: GraphState):
    '''This node will add title based on paragraph or summary(if have)'''
    # # Access the latest human message content as the original text
    text = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage) and not message.content.lower().startswith("feedback title:"):
           text = message.content
           break



    # Access the latest feedback on the title if available
    feedback_title = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage) and message.content.lower().startswith("feedback title:"):
            feedback_title = message.content[len("feedback title:"):].strip()
            break

    prompt_title = f"Create a title based on the following paragraph: \n{text}\n"

    if feedback_title:
        prompt_title += f"\nAdditional requirement for the title: {feedback_title}"

    response = llm.invoke(HumanMessage(content=prompt_title))
    return {"title": response.content}

#=== SUPERVISOR NODE ===
supervisor_prompt = """You are a supervisor agent.
When asked to summarize and add title, call the generate_summary and generate_title tool.
After that, send the result from both tools to val node.
If the message state has a feedback that refines the summary (starts with "Feedback summary:"), then call the tool generate_summary to update the summary based on the feedback.
If the message state has a feedback that refines the title (starts with "Feedback title:"), then call tool generate_title to update the title based on the feedback.
"""

tools = [generate_summary, generate_title]
llm_with_tools= llm.bind_tools(tools, parallel_tool_calls=False)
def supervisor(state: GraphState):
    title = state.get("title", "")
    summary = state.get("summary", "")
    response = llm_with_tools.invoke([SystemMessage(content=supervisor_prompt)] + state["messages"])
    return {"messages": response, "summary":summary, "title": title}

def route_supervisor(state: GraphState) -> Literal["tools", "val"]:
    latest_message = state["messages"][-1]
    # Kiểm tra xem tin nhắn cuối cùng có chứa yêu cầu gọi tool không
    if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
        return "tools"
    else:
        return "val"

#=== VAL NODE ===
def val(state: GraphState):
   summary = state.get("summary","")
   title = state.get("title","")

   print("\n=== Current Result ===")
   print(f"📌 Summary: {summary}")
   print(f"📌 Title: {title}")

   choice = input(
        "\nYou agree with this result?\n"
        "y = Yes\n"
        "n = No\n"
        "Your feedback ('y' for yes, 'n' for no): "
    )
   if choice.lower() == "y":
     return {"messages": [HumanMessage(content="Approved")]}
   elif choice.lower() == "n":
    feedback_summary = input("Feedback summary: ")
    feedback_title = input("Feedback title: ")
    messages_to_add = []
    if feedback_summary:
        messages_to_add.append(HumanMessage(content=f"Feedback summary: {feedback_summary}"))
    if feedback_title:
        messages_to_add.append(HumanMessage(content=f"Feedback title: {feedback_title}"))

    return {"messages": messages_to_add}

def route_val(state: GraphState) -> Literal["supervisor", "__end__"]:
    latest_message_content = state["messages"][-1].content.lower()
    if latest_message_content == "approved":
        return END
    else:
        return "supervisor"

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(GraphState)

builder.add_node("supervisor", supervisor)
builder.add_node("tools", ToolNode(tools))
builder.add_node("val", val)

builder.add_edge(START,"supervisor")
builder.add_conditional_edges("supervisor", route_supervisor)
builder.add_edge("tools", "supervisor") # Tools node always returns to supervisor
builder.add_conditional_edges("val", route_val) # Simplified conditional edges for val

memory=MemorySaver()
app2 = builder.compile(checkpointer=memory)

from langchain_core.messages import HumanMessage
from langgraph.graph import END # Ensure END is imported

if __name__ == "__main__":
  while True:
    user_input = input("Your input: ")
    if user_input.lower() == "exit":
      break
    input_state = {'messages':[HumanMessage(content=user_input)]}
    # The initial state should be just the user message.
    # The supervisor will handle routing based on the state.
    # Include a config dictionary with a configurable key containing the thread_id
    final_state = app2.invoke(input_state, config={"configurable": {"thread_id": "1"}})

    print("\n=== Final State ===")
    print(final_state)