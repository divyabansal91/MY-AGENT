import os
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

app = FastAPI()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat(q: Query):
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=q.message,
    )
    return {"reply": resp.text}

@app.get("/")
def health():
    return {"status": "ok"}
