from fastapi import FastAPI
from pydantic import BaseModel
from hitl_project import app_graph
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
    feedback: str = ""

app = FastAPI()

@app.get("/health", response_model=HealthCheck)
def health_check():
    """
    Endpoint to check the health of the service.
    """
    return HealthCheck(status="OK")

@app.post("/start-summarize/", response_model=SummarizeResponse)
async def start_summarize(request: StartSummarizeRequest):
    initial_state = {"messages": [HumanMessage(content=request.text)]}
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = app_graph.invoke(initial_state, config=thread)
    summary = result["messages"][-1].content
    return SummarizeResponse(summary=summary, thread_id=thread["configurable"]["thread_id"])

@app.post("/submit-feedback/", response_model=SummarizeResponse)
async def submit_feedback(request: SubmitFeedbackRequest):
    
    inputs = {"messages": [HumanMessage(content=f"Refine lại tóm tắt: {request.feedback}")]}
    thread = {"configurable": {"thread_id": request.thread_id}}
    final_state = app_graph.invoke(inputs, config=thread)

    summary_message = final_state["messages"][-1].content

    return SummarizeResponse(summary=summary_message, thread_id=request.thread_id)        
# To run this file, save it as health_check.py and run the following command:
# uvicorn health_check:app --reload
