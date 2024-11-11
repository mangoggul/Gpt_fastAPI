from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .chat_bot import smart_qa
app = FastAPI()

# 요청 모델 정의
class QuestionRequest(BaseModel):
    question: str

@app.post("/api/ask", tags=['AI_ask'])
async def ask_question(request: QuestionRequest):
    try:
        answer = smart_qa(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

