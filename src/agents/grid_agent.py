from pathlib import Path
from typing import TypedDict

import os
import requests
import faiss
import joblib
import numpy as np
import pandas as pd

from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

from src.knowledge_graph.grid_knowledge_graph import get_graph_context


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
    parse_dates=["date"],
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

class AgentState(TypedDict, total=False):

    query: str
    target_date: str

    risk_probability: float
    risk_label: str

    evidence: str
    evidence_date: str

    graph_context: str

    human_review_required: bool

    llm_provider: str
    llm_model: str
    llm_fallback_used: bool

    final_answer: str


# =========================================================
# Tool 1: ML risk prediction
# =========================================================

def risk_prediction_node(
    state: AgentState,
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
    state: AgentState,
):

    query = state["query"]

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
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
        indices[0],
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
# Tool 3: Knowledge Graph
# =========================================================

def knowledge_graph_node(
    state: AgentState,
):

    context = get_graph_context(
        state["query"]
    )

    return {
        "graph_context":
            context
    }


# =========================================================
# Governance / HITL node
# =========================================================

def governance_node(
    state: AgentState,
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
# Deterministic fail-safe
# =========================================================

def deterministic_summary(
    state: AgentState,
) -> str:

    probability_percent = (
        state["risk_probability"] * 100
    )

    return (
        f"The model classifies "
        f"{state['target_date']} as "
        f"{state['risk_label']} with a "
        f"major-outage probability of "
        f"{probability_percent:.1f}%."
    )


# =========================================================
# Qwen through local Ollama
# =========================================================

def call_ollama(
    prompt: str,
) -> tuple[str, str]:

    ollama_url = os.getenv(
        "OLLAMA_URL",
        "http://127.0.0.1:11434",
    )

    model = os.getenv(
        "OLLAMA_MODEL",
        "qwen2.5:3b",
    )

    response = requests.post(
        ollama_url.rstrip("/")
        + "/api/generate",

        json={
            "model": model,
            "prompt": prompt,
            "stream": False,

            "options": {
                "temperature": 0.1,
            },
        },

        timeout=float(
            os.getenv(
                "OLLAMA_TIMEOUT_SECONDS",
                "10",
            )
        ),
    )

    response.raise_for_status()

    generated = (
        response.json()
        .get("response", "")
        .strip()
    )

    if not generated:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return generated, model


# =========================================================
# OpenRouter LLM
# =========================================================

def call_openrouter(
    prompt: str,
) -> tuple[str, str]:

    api_key = os.getenv(
        "OPENROUTER_API_KEY",
        "",
    ).strip()

    if (
        not api_key
        or api_key == "YOUR_KEY_HERE"
    ):
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing."
        )

    model = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free",
    )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",

        headers={
            "Authorization":
                f"Bearer {api_key}",

            "Content-Type":
                "application/json",

            "X-Title":
                "GridSentinel AI",
        },

        json={
            "model": model,

            "messages": [
                {
                    "role": "system",

                    "content": (
                        "You are GridSentinel AI, "
                        "a safety-conscious utility "
                        "risk assessment assistant. "
                        "Use only supplied model output, "
                        "retrieved evidence, and "
                        "structured knowledge graph context. "
                        "Do not invent facts. "
                        "Do not modify numbers. "
                        "Do not provide autonomous "
                        "safety-critical commands."
                    ),
                },

                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            "temperature": 0.1,
            "max_tokens": 220,
        },

        timeout=float(
            os.getenv(
                "OPENROUTER_TIMEOUT_SECONDS",
                "30",
            )
        ),
    )

    response.raise_for_status()

    data = response.json()

    generated = (
        data["choices"][0]
        ["message"]
        ["content"]
        .strip()
    )

    if not generated:
        raise RuntimeError(
            "OpenRouter returned an empty response."
        )

    return generated, model


# =========================================================
# LLM provider routing
# =========================================================

def generate_llm_summary(
    state: AgentState,
    prompt: str,
):

    provider = os.getenv(
        "LLM_PROVIDER",
        "auto",
    ).strip().lower()

    if provider == "auto":

        if os.getenv("K_SERVICE"):
            provider = "openrouter"
        else:
            provider = "ollama"

    try:

        if provider == "openrouter":

            text, model = call_openrouter(
                prompt
            )

            return {
                "text":
                    text,

                "provider":
                    "OpenRouter",

                "model":
                    model,

                "fallback":
                    False,
            }

        if provider == "ollama":

            text, model = call_ollama(
                prompt
            )

            return {
                "text":
                    text,

                "provider":
                    "Ollama",

                "model":
                    model,

                "fallback":
                    False,
            }

        raise ValueError(
            f"Unsupported LLM_PROVIDER: "
            f"{provider}"
        )

    except (
        requests.RequestException,
        RuntimeError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ):

        return {
            "text":
                deterministic_summary(
                    state
                ),

            "provider":
                "Deterministic fallback",

            "model":
                "none",

            "fallback":
                True,
        }


# =========================================================
# Final response node
# =========================================================

def answer_node(
    state: AgentState,
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
User question:
{state["query"]}

Target date:
{state["target_date"]}

ML risk classification:
{state["risk_label"]}

Major-outage probability:
{probability_percent:.1f}%

Retrieved trusted evidence:
{state["evidence"]}

Structured knowledge graph context:
{state["graph_context"]}

Write a concise 1-2 sentence risk assessment.

Rules:
- Use only the information above.
- Do not invent facts.
- Do not change numeric values.
- Do not provide autonomous operational commands.
- Do not claim causal relationships.
- If risk is high, you may state that human engineering review is required.
"""

    llm_result = generate_llm_summary(
        state,
        prompt,
    )

    llm_summary = (
        llm_result["text"]
    )

    provider = (
        llm_result["provider"]
    )

    model = (
        llm_result["model"]
    )

    fallback = (
        llm_result["fallback"]
    )

    final_answer = f"""
1. Risk Assessment
{llm_summary}

2. Evidence
{state["evidence"]}

3. Knowledge Graph Context
{state["graph_context"]}

4. Governance
Governance status: {governance_text}

5. LLM Runtime
Provider: {provider}
Model: {model}
Fallback used: {fallback}
""".strip()

    return {
        "llm_provider":
            provider,

        "llm_model":
            model,

        "llm_fallback_used":
            fallback,

        "final_answer":
            final_answer,
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
    "knowledge_graph",
    knowledge_graph_node,
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
    "knowledge_graph",
)


graph.add_edge(
    "knowledge_graph",
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
# Local test
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