import json
import time
from datetime import datetime
from pathlib import Path


LOG_DIR = Path(
    "src/evaluation/logs"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = LOG_DIR / "agent_runtime.jsonl"


def log_agent_run(
    *,
    query: str,
    target_date: str,
    risk_probability: float,
    risk_label: str,
    evidence_date: str,
    human_review_required: bool,
    llm_provider: str,
    llm_model: str,
    llm_fallback_used: bool,
    latency_seconds: float,
):

    record = {
        "timestamp":
            datetime.utcnow().isoformat(),

        "query":
            query,

        "target_date":
            target_date,

        "risk_probability":
            risk_probability,

        "risk_label":
            risk_label,

        "evidence_date":
            evidence_date,

        "human_review_required":
            human_review_required,

        "llm_provider":
            llm_provider,

        "llm_model":
            llm_model,

        "llm_fallback_used":
            llm_fallback_used,

        "latency_seconds":
            latency_seconds,
    }

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(record)
            + "\n"
        )