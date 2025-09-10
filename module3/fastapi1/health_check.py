from fastapi import FastAPI
from pydantic import BaseModel
from hitl_project import app_graph, human_feedback
from langchain_core.messages import HumanMessage
import uuid

class HealthCheck(BaseModel):
    status: str = "OK"

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

@app.get("/health", response_model=HealthCheck)
def health_check():
    """
    Endpoint to check the health of the service.
    """
    return HealthCheck(status="OK")

@app.post("/start-summarize/", response_model=SummarizeResponse)
async def start_summarize(request: StartSummarizeRequest):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    # Initial state with the document text as the first message
    initial_state = {"messages": [HumanMessage(content=request.text)], "approved": False}
    result = app_graph.invoke(initial_state, config=config)
    summary = result["messages"][-1].content
    return SummarizeResponse(summary=summary, thread_id=config["configurable"]["thread_id"])

@app.post("/submit-feedback/", response_model=SummarizeResponse)
async def submit_feedback(request: SubmitFeedbackRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Get the current state to pass to human_feedback
    current_state = app_graph.get_state(config=config)

    # Manually call the human_feedback function with the provided parameters
    feedback_output = human_feedback(current_state.values, request.approved, request.feedback_text)
    
    # Invoke the graph with the output of human_feedback
    # The graph will decide the next step (summarize again or save) based on the 'approved' flag
    result = app_graph.invoke(feedback_output, config=config)

    # After invoking, the graph would have either re-summarized or saved.
    # We need to get the latest summary from the state.
    final_state = app_graph.get_state(config=config)
    summary_message = final_state.values["messages"][-1].content

    return SummarizeResponse(summary=summary_message, thread_id=request.thread_id)        
# To run this file, save it as health_check.py and run the following command:
# uvicorn health_check:app --reload
