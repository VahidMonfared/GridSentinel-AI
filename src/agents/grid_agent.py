from pathlib import Path
from typing import TypedDict

import requests
import faiss
import joblib
import numpy as np
import pandas as pd
import os

from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer


# =========================================================
# Files
# =========================================================

DATA_FILE = Path(
    "data/processed/harris_model_dataset_2024.csv"
)

MODEL_FILE = Path(
    "src/models/artifacts/best_model.joblib"
)

RAG_DIR = Path(
    "src/rag/artifacts"
)

FAISS_FILE = RAG_DIR / "grid_knowledge.faiss"

DOCS_FILE = RAG_DIR / "knowledge_documents.joblib"


# =========================================================
# Load shared resources once
# =========================================================

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

risk_model = joblib.load(
    MODEL_FILE
)

faiss_index = faiss.read_index(
    str(FAISS_FILE)
)

documents = joblib.load(
    DOCS_FILE
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# Features used by final binary model
# =========================================================

FEATURES = [
    "storm_event",
    "storm_count",

    "is_hurricane",
    "is_derecho",
    "is_tornado",
    "is_thunderstorm_wind",
    "is_flood",

    "storm_recent_7d",
    "storm_count_7d",

    "is_hurricane_recent_7d",
    "is_derecho_recent_7d",
    "is_tornado_recent_7d",
    "is_thunderstorm_wind_recent_7d",
    "is_flood_recent_7d",

    "month",
    "day_of_year",
    "day_of_week",

    "outage_lag_1d",
    "outage_mean_7d",
    "outage_max_7d",
]


# =========================================================
# Agent state
# =========================================================

class AgentState(TypedDict):

    query: str
    target_date: str

    risk_probability: float
    risk_label: str

    evidence: str
    evidence_date: str

    human_review_required: bool

    final_answer: str


# =========================================================
# Tool 1: ML risk prediction
# =========================================================

def risk_prediction_node(
    state: AgentState
):

    target_date = pd.Timestamp(
        state["target_date"]
    )

    row = df[
        df["date"] == target_date
    ]

    if row.empty:

        raise ValueError(
            f"No data found for "
            f"{state['target_date']}"
        )

    probability = risk_model.predict_proba(
        row[FEATURES]
    )[0, 1]

    label = (
        "HIGH RISK"
        if probability >= 0.50
        else "LOW RISK"
    )

    return {
        "risk_probability":
            float(probability),

        "risk_label":
            label,
    }


# =========================================================
# Tool 2: Hybrid RAG retrieval
# =========================================================

def evidence_retrieval_node(
    state: AgentState
):

    query = state["query"]

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = faiss_index.search(
        query_embedding,
        20,
    )

    candidates = []

    query_lower = query.lower()

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        doc = documents[idx]

        text_lower = (
            doc["text"].lower()
        )

        bonus = 0.0

        if (
            "hurricane" in query_lower
            and
            "hurricane indicator:\n1"
            in text_lower
        ):
            bonus += 0.30

        if (
            "beryl" in query_lower
            and
            "beryl" in text_lower
        ):
            bonus += 0.40

        if (
            "tornado" in query_lower
            and
            "tornado indicator:\n1"
            in text_lower
        ):
            bonus += 0.30

        if (
            "flood" in query_lower
            and
            "flood indicator:\n1"
            in text_lower
        ):
            bonus += 0.30

        candidates.append(
            {
                "score":
                    float(score + bonus),

                "doc":
                    doc,
            }
        )

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True,
    )

    best = candidates[0]

    return {
        "evidence":
            best["doc"]["text"],

        "evidence_date":
            best["doc"]["date"],
    }


# =========================================================
# Governance / HITL node
# =========================================================

def governance_node(
    state: AgentState
):

    probability = (
        state["risk_probability"]
    )

    require_review = (
        probability >= 0.50
    )

    return {
        "human_review_required":
            require_review
    }


# =========================================================
# Final response node
# =========================================================

def answer_node(
    state: AgentState
):

    probability_percent = (
        state["risk_probability"] * 100
    )

    if state["human_review_required"]:
        governance_text = (
            "Human engineering review required."
        )
    else:
        governance_text = (
            "Routine monitoring."
        )

    prompt = f"""
You are GridSentinel AI.

Use ONLY the information below.

Do not invent facts.
Do not change numbers.
Do not rename evidence fields.
Do not provide operational recommendations.

User question:
{state["query"]}

Target date:
{state["target_date"]}

ML risk classification:
{state["risk_label"]}

Major-outage probability:
{probability_percent:.1f}%

Write ONLY a short 1-2 sentence risk assessment.
Do not reproduce the evidence.
"""

    response = requests.post(
        os.getenv(
            "OLLAMA_URL",
            "http://127.0.0.1:11434"
        ) + "/api/generate",
        json={
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            },
        },
        timeout=120,
    )

    response.raise_for_status()

    llm_summary = response.json()[
        "response"
    ].strip()

    final_answer = f"""
1. Risk Assessment
{llm_summary}

2. Evidence
{state["evidence"]}

3. Governance
Governance status: {governance_text}
""".strip()

    return {
        "final_answer": final_answer
    }

# =========================================================
# Build LangGraph
# =========================================================

graph = StateGraph(
    AgentState
)

graph.add_node(
    "risk_prediction",
    risk_prediction_node,
)

graph.add_node(
    "retrieve_evidence",
    evidence_retrieval_node,
)

graph.add_node(
    "governance",
    governance_node,
)

graph.add_node(
    "answer",
    answer_node,
)


graph.set_entry_point(
    "risk_prediction"
)

graph.add_edge(
    "risk_prediction",
    "retrieve_evidence",
)

graph.add_edge(
    "retrieve_evidence",
    "governance",
)

graph.add_edge(
    "governance",
    "answer",
)

graph.add_edge(
    "answer",
    END,
)


agent = graph.compile()


# =========================================================
# Test Beryl
# =========================================================

if __name__ == "__main__":

    result = agent.invoke(
        {
            "query":
                "What happened during "
                "Hurricane Beryl and "
                "how severe was the outage?",

            "target_date":
                "2024-07-08",
        }
    )

    print(
        "\n========================="
    )

    print(
        "GridSentinel Agent Result"
    )

    print(
        "=========================\n"
    )

    print(
        result["final_answer"]
    )