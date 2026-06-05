from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_endpoint(body: TextInput):
    return predict(body.text)

@app.get("/health")
def health():
    return {"status": "ok"}