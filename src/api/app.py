from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.grid_agent import agent


app = FastAPI(
    title="GridSentinel AI API",
    version="1.0.0",
    description=(
        "Governed agentic AI for grid outage risk assessment."
    ),
)


class AssessmentRequest(BaseModel):
    query: str
    target_date: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "GridSentinel AI",
    }


@app.post("/assess")
def assess(request: AssessmentRequest):

    try:
        result = agent.invoke(
            {
                "query": request.query,
                "target_date": request.target_date,
            }
        )

        return {
            "target_date": request.target_date,
            "risk_probability": result[
                "risk_probability"
            ],
            "risk_label": result[
                "risk_label"
            ],
            "evidence_date": result[
                "evidence_date"
            ],
            "human_review_required": result[
                "human_review_required"
            ],
            "answer": result[
                "final_answer"
            ],
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )