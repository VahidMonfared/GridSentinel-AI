from pathlib import Path
from typing import TypedDict

import os
import re

import requests
import faiss
import joblib
import numpy as np
import pandas as pd

from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

from src.knowledge_graph.grid_knowledge_graph import (
    get_graph_context,
)


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

FAISS_FILE = (
    RAG_DIR
    / "grid_knowledge.faiss"
)

DOCS_FILE = (
    RAG_DIR
    / "knowledge_documents.joblib"
)


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

class AgentState(
    TypedDict,
    total=False,
):

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
        df["date"]
        == target_date
    ]

    if row.empty:

        raise ValueError(
            f"No data found for "
            f"{state['target_date']}"
        )

    probability = (
        risk_model.predict_proba(
            row[FEATURES]
        )[0, 1]
    )

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

    scores, indices = (
        faiss_index.search(
            query_embedding,
            20,
        )
    )

    candidates = []

    query_lower = (
        query.lower()
    )

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
            "hurricane"
            in query_lower
            and
            "hurricane indicator:\n1"
            in text_lower
        ):
            bonus += 0.30

        if (
            "beryl"
            in query_lower
            and
            "beryl"
            in text_lower
        ):
            bonus += 0.40

        if (
            "derecho"
            in query_lower
            and
            "derecho indicator:\n1"
            in text_lower
        ):
            bonus += 0.40

        if (
            "tornado"
            in query_lower
            and
            "tornado indicator:\n1"
            in text_lower
        ):
            bonus += 0.30

        if (
            "flood"
            in query_lower
            and
            "flood indicator:\n1"
            in text_lower
        ):
            bonus += 0.30

        candidates.append(
            {
                "score":
                    float(
                        score
                        + bonus
                    ),

                "doc":
                    doc,
            }
        )

    candidates = sorted(
        candidates,
        key=lambda item:
            item["score"],
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

    context = (
        get_graph_context(
            state["query"]
        )
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
        state[
            "risk_probability"
        ]
    )

    require_review = (
        probability >= 0.50
    )

    return {
        "human_review_required":
            require_review
    }


# =========================================================
# Evidence utilities
# =========================================================

def extract_evidence_value(
    evidence: str,
    label: str,
):

    lines = [
        line.strip()
        for line
        in evidence.splitlines()
        if line.strip()
    ]

    for index, line in enumerate(
        lines
    ):

        if (
            line.lower()
            == f"{label.lower()}:"
        ):

            if (
                index + 1
                < len(lines)
            ):
                return lines[
                    index + 1
                ]

    return None


def format_integer(
    value,
):

    try:

        return (
            f"{int(float(value)):,}"
        )

    except (
        TypeError,
        ValueError,
    ):

        return str(value)


def format_percent(
    value,
):

    try:

        return (
            f"{float(value) * 100:.1f}%"
        )

    except (
        TypeError,
        ValueError,
    ):

        return str(value)


# =========================================================
# Deterministic client-safe risk summary
# =========================================================

def build_safe_risk_summary(
    state: AgentState,
):

    evidence = (
        state["evidence"]
    )

    probability_percent = (
        state[
            "risk_probability"
        ]
        * 100
    )

    location = (
        extract_evidence_value(
            evidence,
            "Location",
        )
        or "Harris County, Texas"
    )

    peak_customers = (
        extract_evidence_value(
            evidence,
            "Peak customers without power",
        )
    )

    peak_rate = (
        extract_evidence_value(
            evidence,
            "Peak outage rate",
        )
    )

    storm_type = (
        extract_evidence_value(
            evidence,
            "Storm types",
        )
    )

    if (
        storm_type
        and
        storm_type.lower()
        not in {
            "none",
            "nan",
            "0",
        }
    ):

        opening = (
            f"{storm_type} was associated "
            f"with significant grid disruption "
            f"in {location}."
        )

    else:

        opening = (
            f"The assessment for "
            f"{state['target_date']} "
            f"in {location} indicates "
            f"{state['risk_label'].lower()}."
        )

    evidence_parts = []

    if peak_customers:

        evidence_parts.append(
            f"{format_integer(peak_customers)} "
            f"customers were without power "
            f"at the daily peak"
        )

    if peak_rate:

        evidence_parts.append(
            f"the peak outage rate reached "
            f"{format_percent(peak_rate)}"
        )

    if evidence_parts:

        evidence_sentence = (
            "Observed evidence shows that "
            + " and ".join(
                evidence_parts
            )
            + "."
        )

    else:

        evidence_sentence = ""

    model_sentence = (
        f"The model estimates a "
        f"{probability_percent:.1f}% "
        f"probability of a major outage."
    )

    if state[
        "human_review_required"
    ]:

        governance_sentence = (
            "Qualified human engineering "
            "review is required before any "
            "safety-critical action."
        )

    else:

        governance_sentence = (
            "The case remains within "
            "routine monitoring thresholds."
        )

    summary = " ".join(
        part
        for part in [
            opening,
            evidence_sentence,
            model_sentence,
            governance_sentence,
        ]
        if part
    )

    return summary


# =========================================================
# Deterministic fail-safe
# =========================================================

def deterministic_context_note(
    state: AgentState,
) -> str:

    if state[
        "human_review_required"
    ]:

        return (
            "The retrieved evidence and "
            "structured model output should "
            "be reviewed by a qualified "
            "engineer before operational use."
        )

    return (
        "The retrieved evidence is consistent "
        "with routine monitoring rather than "
        "automatic escalation."
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
            "model":
                model,

            "prompt":
                prompt,

            "stream":
                False,

            "options": {
                "temperature":
                    0.1,
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
        .get(
            "response",
            "",
        )
        .strip()
    )

    if not generated:

        raise RuntimeError(
            "Ollama returned "
            "an empty response."
        )

    return (
        generated,
        model,
    )


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
        or
        api_key
        == "YOUR_KEY_HERE"
    ):

        raise RuntimeError(
            "OPENROUTER_API_KEY "
            "is missing."
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
            "model":
                model,

            "messages": [
                {
                    "role":
                        "system",

                    "content": (
                        "You are GridSentinel AI. "
                        "Return only one short "
                        "plain-English qualitative "
                        "sentence. "
                        "Do not include numbers. "
                        "Do not show reasoning steps. "
                        "Do not use phrases such as "
                        "'thinking process', "
                        "'step 1', 'analysis', "
                        "or 'chain of thought'. "
                        "Do not invent facts. "
                        "Do not issue operational "
                        "commands."
                    ),
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt,
                },
            ],

            "temperature":
                0.1,

            "max_tokens":
                80,
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
            "OpenRouter returned "
            "an empty response."
        )

    return (
        generated,
        model,
    )


# =========================================================
# Client-output LLM guardrail
# =========================================================

def validate_llm_note(
    text: str,
):

    lowered = (
        text.lower()
    )

    forbidden_phrases = [
        "thinking process",
        "chain of thought",
        "step 1",
        "step 2",
        "analyze user",
        "analysis:",
        "reasoning:",
        "identify key",
    ]

    if any(
        phrase in lowered
        for phrase
        in forbidden_phrases
    ):

        return False

    if re.search(
        r"\d",
        text,
    ):

        return False

    if len(
        text.split()
    ) > 45:

        return False

    return True


# =========================================================
# LLM provider routing
# =========================================================

def generate_llm_context_note(
    state: AgentState,
    prompt: str,
):

    provider = os.getenv(
        "LLM_PROVIDER",
        "auto",
    ).strip().lower()

    if provider == "auto":

        if os.getenv(
            "K_SERVICE"
        ):

            provider = (
                "openrouter"
            )

        else:

            provider = (
                "ollama"
            )

    try:

        if (
            provider
            == "openrouter"
        ):

            text, model = (
                call_openrouter(
                    prompt
                )
            )

            provider_name = (
                "OpenRouter"
            )

        elif (
            provider
            == "ollama"
        ):

            text, model = (
                call_ollama(
                    prompt
                )
            )

            provider_name = (
                "Ollama"
            )

        else:

            raise ValueError(
                f"Unsupported "
                f"LLM_PROVIDER: "
                f"{provider}"
            )

        if not validate_llm_note(
            text
        ):

            return {
                "text":
                    deterministic_context_note(
                        state
                    ),

                "provider":
                    provider_name,

                "model":
                    model,

                "fallback":
                    True,
            }

        return {
            "text":
                text,

            "provider":
                provider_name,

            "model":
                model,

            "fallback":
                False,
        }

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
                deterministic_context_note(
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

    safe_summary = (
        build_safe_risk_summary(
            state
        )
    )

    prompt = f"""
User question:
{state["query"]}

Risk label:
{state["risk_label"]}

Knowledge graph context:
{state["graph_context"]}

Write one short qualitative sentence
that adds context for the client.

Do not include any numbers.
Do not repeat the risk probability.
Do not show your reasoning process.
Do not issue operational commands.
"""

    llm_result = (
        generate_llm_context_note(
            state,
            prompt,
        )
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

    if state[
        "human_review_required"
    ]:

        governance_text = (
            "Human engineering "
            "review required."
        )

    else:

        governance_text = (
            "Routine monitoring."
        )

    final_answer = f"""
1. Risk Assessment
{safe_summary}

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