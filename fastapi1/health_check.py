from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.graph import graph
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AnyMessage

class HealthCheck(BaseModel):
    status: str = "OK"

class Message(BaseModel):
    messages: str

app = FastAPI()

@app.get("/health", response_model=HealthCheck)
def health_check():
    """
    Endpoint to check the health of the service.
    """
    return HealthCheck(status="OK")

@app.post("/messages", response_model=Message)
def dev_test(messages: Message):

    result = graph.invoke({"messages": [HumanMessage(content="Hello, how are you?")]})

    return result
# To run this file, save it as health_check.py and run the following command:
# uvicorn health_check:app --reload
