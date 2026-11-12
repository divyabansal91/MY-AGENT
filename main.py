import os
from fastapi import FastAPI
from pydantic import BaseModel
import anthropic

app = FastAPI()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat(q: Query):
    resp = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": q.message}],
    )
    return {"reply": resp.content[0].text}

@app.get("/")
def health():
    return {"status": "ok"}
