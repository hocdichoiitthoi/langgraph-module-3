from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Annotated, TypedDict
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import uuid

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    approved: bool


llm = AzureChatOpenAI(
    model="gpt-4.1",
    api_version="2025-01-01-preview",
    temperature=0)


def summarize_doc(state: State):
    text = state["messages"][-1].content
    feedback = ""
    if len(state["messages"]) > 1:
        feedbacks = [m.content for m in state["messages"][1:]]
        feedback = "\n".join(feedbacks)

    prompt = f"Tóm tắt văn bản sao cho vẫn nắm được ý chính:\n\n{text}\n\n"
    if feedback:
        prompt += f"Yêu cầu chỉnh sửa bổ sung: {feedback}"

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


def save_summary(state: State):
    print("\n✅ Tóm tắt cuối cùng được lưu!")
    print(state["messages"][-1].content)
    return {}


def decide_next(state: State):
    if state["approved"] == True:
        return "save_summary"
    else:
        return "human_feedback"


# Node Human: duyệt hoặc chỉnh sửa
def human_feedback(state: State, approved: bool, feedback_text: str = ""):
    if approved:
        return {"approved": True}
    else:
        return {"messages": [HumanMessage(content=f"Refine lại tóm tắt: {feedback_text}")], "approved": False}


# Node Save: lưu tóm tắt


class StartSummarizeRequest(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary: str
    thread_id: str


class SubmitFeedbackRequest(BaseModel):
    thread_id: str
    approved: bool
    feedback_text: str = ""


app = FastAPI()

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


@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI!"}


@app.post("/start-summarize/")
async def start_summarize(request: StartSummarizeRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"messages": [HumanMessage(content=request.text)], "approved": False}

    async def generate_summary_stream():
        current_summary_content = ""
        async for s in app_graph.stream(initial_state, config=config):
            # Extract the message content from the latest message in the state
            if "messages" in s and s["messages"]:
                # Assuming the last message is the most relevant for streaming progress
                latest_message_content = s["messages"][-1].content

                # If the content has changed, yield the difference
                if latest_message_content and latest_message_content != current_summary_content:
                    new_chunk = latest_message_content[len(current_summary_content):]
                    if new_chunk:
                        yield f"data: {{'thread_id': '{thread_id}', 'summary_chunk': {new_chunk!r}}}\n\n"
                    current_summary_content = latest_message_content

    return StreamingResponse(generate_summary_stream(), media_type="text/event-stream")


@app.post("/submit-feedback/", response_model=SummarizeResponse)
async def submit_feedback(request: SubmitFeedbackRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    # We need to call the human_feedback node directly as it expects specific parameters
    # and then invoke the graph from that point.
    # The graph will then decide to summarize again or save.
    
    # Get the current state to pass to human_feedback, if needed, though for this node
    # the state isn't directly used for its logic but for the graph's overall flow.
    current_state = app_graph.get_state(config=config)

    # Manually call the human_feedback function with the provided parameters
    feedback_output = human_feedback(current_state.values, request.approved, request.feedback_text)
    
    # Invoke the graph from the 'feedback' node with the output of human_feedback
    # and let it decide the next step (summarize again or save)
    # The 'feedback' node needs to be explicitly set as the next node if we are manually calling it.
    # However, the graph is set up to go from 'feedback' to 'summarize' or 'save' via 'decide_next'
    # So, we just need to update the state and continue.

    # If approved, the graph goes to save. If not, it goes to summarize again.
    if request.approved:
        # Directly transition to save if approved
        result = app_graph.invoke(feedback_output, config=config, name="feedback")
    else:
        # If not approved, the feedback_output contains the message for refinement
        # and the graph will transition from feedback to summarize via decide_next.
        result = app_graph.invoke(feedback_output, config=config, name="feedback")

    # After invoking, the graph would have either re-summarized or saved.
    # We need to get the latest summary from the state.
    final_state = app_graph.get_state(config=config)
    summary_message = final_state.values["messages"][-1].content

    return SummarizeResponse(summary=summary_message, thread_id=request.thread_id)
