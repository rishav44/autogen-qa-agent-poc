
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agents.question_answer_agent import QuestionAnswerAgent


app = FastAPI(
    title="AutoGen Question Answer Agent",
    description="Generate questions, detailed answers and verified references.",
    version="1.0.0",
)


class QARequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200)
    complexity: Literal["Beginner", "Intermediate", "Advanced"]
    question_count: int = Field(..., ge=1, le=10)


class QAResponse(BaseModel):
    topic: str
    complexity: str
    question_count: int
    result: str


@app.get("/")
async def home():
    return {
        "message": "AutoGen Question Answer Agent API is running",
        "documentation": "/docs",
    }


@app.post("/generate", response_model=QAResponse)
async def generate_questions(request: QARequest):

    qa_agent = QuestionAnswerAgent()

    try:
        result = await qa_agent.generate_response(
            topic=request.topic,
            complexity=request.complexity,
            question_count=request.question_count,
        )

        return QAResponse(
            topic=request.topic,
            complexity=request.complexity,
            question_count=request.question_count,
            result=result,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Question generation failed: {str(error)}",
        )

    finally:
        await qa_agent.model_client.close()
