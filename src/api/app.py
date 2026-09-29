import time
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.agents.grid_agent import agent
from src.evaluation.observability import log_agent_run
from src.models.reasoning_explainer import (
    explain_prediction,
)


# ==================================================
# App
# ==================================================

app = FastAPI(
    title="GridSentinel AI",
    version="1.3.0",
    description=(
        "Governed agentic AI for extreme-weather "
        "grid outage risk assessment."
    ),
)


# ==================================================
# Historical validation artifacts
# ==================================================

DATA_FILE = Path(
    "data/processed/harris_model_dataset_2024.csv"
)

MODEL_METADATA_FILE = Path(
    "src/models/artifacts/model_metadata.joblib"
)

MODEL_RESULTS_FILE = Path(
    "src/models/artifacts/model_comparison.csv"
)


validation_df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"],
)

validation_df = (
    validation_df
    .sort_values("date")
    .reset_index(drop=True)
)


model_metadata = joblib.load(
    MODEL_METADATA_FILE
)

major_outage_threshold = float(
    model_metadata["threshold"]
)

split_date = pd.Timestamp(
    model_metadata["split_date"]
)


model_results = pd.read_csv(
    MODEL_RESULTS_FILE
)

best_results = (
    model_results[
        model_results["model"]
        == "logistic_regression"
    ]
    .iloc[0]
)


# Number of true major-outage days
# in the held-out Jul-Dec test period.
test_validation_df = validation_df[
    validation_df["date"] >= split_date
].copy()

test_validation_df[
    "actual_major_outage"
] = (
    test_validation_df[
        "daily_peak_outage"
    ]
    >= major_outage_threshold
).astype(int)

total_test_major_outages = int(
    test_validation_df[
        "actual_major_outage"
    ].sum()
)

# Recall = TP / all actual positives.
# Derive the number detected from the
# stored evaluation metric.
detected_test_major_outages = int(
    round(
        float(
            best_results["recall"]
        )
        * total_test_major_outages
    )
)


# ==================================================
# Request model
# ==================================================

class AssessmentRequest(BaseModel):
    query: str
    target_date: str


# ==================================================
# Helpers
# ==================================================

def extract_risk_summary(
    final_answer: str,
):

    text = final_answer

    start_marker = "1. Risk Assessment"
    end_marker = "2. Evidence"

    if start_marker in text:
        text = text.split(
            start_marker,
            1,
        )[1]

    if end_marker in text:
        text = text.split(
            end_marker,
            1,
        )[0]

    return text.strip()


def parse_evidence(
    evidence: str,
):

    lines = [
        line.strip()
        for line in evidence.splitlines()
        if line.strip()
    ]

    parsed = {}

    index = 0

    while index < len(lines):

        line = lines[index]

        if line.endswith(":"):

            key = line[:-1]

            if index + 1 < len(lines):

                parsed[key] = (
                    lines[index + 1]
                )

                index += 2
                continue

        index += 1

    return parsed


def get_historical_validation(
    target_date: str,
    risk_probability: float,
):

    target = pd.Timestamp(
        target_date
    )

    row = validation_df[
        validation_df["date"] == target
    ]

    if row.empty:

        return None

    row = row.iloc[0]

    actual_peak = float(
        row["daily_peak_outage"]
    )

    actual_major_outage = bool(
        actual_peak
        >= major_outage_threshold
    )

    predicted_major_outage = bool(
        risk_probability
        >= 0.50
    )

    is_test_period = bool(
        target >= split_date
    )

    return {
        "period":
            (
                "Held-out test"
                if is_test_period
                else "Training period"
            ),

        "is_out_of_sample":
            is_test_period,

        "predicted_label":
            (
                "HIGH RISK"
                if predicted_major_outage
                else "LOW RISK"
            ),

        "actual_label":
            (
                "MAJOR OUTAGE"
                if actual_major_outage
                else "NO MAJOR OUTAGE"
            ),

        "correct":
            (
                predicted_major_outage
                == actual_major_outage
            ),

        "actual_peak_customers_out":
            int(
                round(
                    actual_peak
                )
            ),

        "major_outage_threshold":
            float(
                major_outage_threshold
            ),
    }


# ==================================================
# API
# ==================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "GridSentinel AI",
    }


@app.post("/assess")
def assess(
    request: AssessmentRequest,
):

    start_time = time.perf_counter()

    try:

        result = agent.invoke(
            {
                "query":
                    request.query,

                "target_date":
                    request.target_date,
            }
        )

        reasoning = explain_prediction(
            request.target_date
        )

        historical_validation = (
            get_historical_validation(
                request.target_date,
                result[
                    "risk_probability"
                ],
            )
        )

        latency_seconds = (
            time.perf_counter()
            - start_time
        )

        log_agent_run(
            query=request.query,

            target_date=
                request.target_date,

            risk_probability=
                result[
                    "risk_probability"
                ],

            risk_label=
                result[
                    "risk_label"
                ],

            evidence_date=
                result[
                    "evidence_date"
                ],

            human_review_required=
                result[
                    "human_review_required"
                ],

            llm_provider=
                result.get(
                    "llm_provider",
                    "unknown",
                ),

            llm_model=
                result.get(
                    "llm_model",
                    "unknown",
                ),

            llm_fallback_used=
                result.get(
                    "llm_fallback_used",
                    False,
                ),

            latency_seconds=
                latency_seconds,
        )

        evidence = parse_evidence(
            result.get(
                "evidence",
                "",
            )
        )

        return {
            "target_date":
                request.target_date,

            "risk_probability":
                result[
                    "risk_probability"
                ],

            "risk_label":
                result[
                    "risk_label"
                ],

            "human_review_required":
                result[
                    "human_review_required"
                ],

            "risk_summary":
                extract_risk_summary(
                    result[
                        "final_answer"
                    ]
                ),

            "evidence":
                evidence,

            "graph_context":
                result.get(
                    "graph_context",
                    "",
                ),

            "reasoning":
                reasoning,

            "historical_validation":
                historical_validation,

            "test_performance": {
                "period":
                    "July-December 2024",

                "roc_auc":
                    float(
                        best_results[
                            "roc_auc"
                        ]
                    ),

                "pr_auc":
                    float(
                        best_results[
                            "pr_auc"
                        ]
                    ),

                "precision":
                    float(
                        best_results[
                            "precision"
                        ]
                    ),

                "recall":
                    float(
                        best_results[
                            "recall"
                        ]
                    ),

                "f1":
                    float(
                        best_results[
                            "f1"
                        ]
                    ),

                "brier_score":
                    float(
                        best_results[
                            "brier_score"
                        ]
                    ),

                "detected_major_outages":
                    detected_test_major_outages,

                "total_major_outages":
                    total_test_major_outages,
            },

            "llm_provider":
                result.get(
                    "llm_provider",
                    "unknown",
                ),

            "llm_model":
                result.get(
                    "llm_model",
                    "unknown",
                ),

            "llm_fallback_used":
                result.get(
                    "llm_fallback_used",
                    False,
                ),

            "latency_seconds":
                latency_seconds,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ==================================================
# UI
# ==================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():

    return """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
    GridSentinel AI
</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background: #f5f7fa;
    color: #182234;
}

.page {
    max-width: 980px;
    margin: 0 auto;
    padding: 42px 22px;
}

.header {
    margin-bottom: 26px;
}

.brand {
    font-size: 32px;
    font-weight: 780;
    letter-spacing: -0.9px;
}

.subtitle {
    margin-top: 7px;
    color: #697487;
    line-height: 1.5;
}

.badge {
    display: inline-block;
    margin-top: 13px;
    padding: 6px 11px;
    border-radius: 999px;
    background: #e9eef8;
    color: #344765;
    font-size: 12px;
    font-weight: 650;
}

.card {
    background: white;
    border: 1px solid #e1e5eb;
    border-radius: 17px;
    padding: 24px;

    box-shadow:
        0 8px 28px
        rgba(20,30,50,0.05);
}

.field {
    margin-bottom: 17px;
}

label {
    display: block;
    margin-bottom: 7px;
    font-size: 13px;
    font-weight: 700;
}

input,
textarea {
    width: 100%;
    border: 1px solid #ccd3dd;
    border-radius: 10px;
    padding: 12px 14px;
    font: inherit;
}

textarea {
    min-height: 105px;
    resize: vertical;
}

button {
    width: 100%;
    border: 0;
    border-radius: 10px;
    padding: 13px;
    background: #172033;
    color: white;
    font-size: 15px;
    font-weight: 720;
    cursor: pointer;
}

button:disabled {
    opacity: 0.55;
}

.result {
    display: none;
    margin-top: 24px;
}

.metrics {
    display: grid;
    grid-template-columns:
        repeat(4, 1fr);
    gap: 11px;
}

.metric {
    background: #f8fafc;
    border: 1px solid #e2e6ec;
    border-radius: 12px;
    padding: 14px;
}

.metric-name {
    color: #7a8495;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.metric-value {
    margin-top: 5px;
    font-size: 18px;
    font-weight: 780;
}

.section {
    margin-top: 16px;
    border: 1px solid #e2e6ec;
    border-radius: 13px;
    padding: 18px;
    background: white;
}

.section-title {
    margin-bottom: 10px;
    font-size: 15px;
    font-weight: 780;
}

.summary {
    color: #354156;
    line-height: 1.65;
    font-size: 14px;
}

.validation-highlight {
    background: #f7faf8;
    border: 1px solid #d8e7de;
}

.validation-grid {
    display: grid;
    grid-template-columns:
        repeat(3, 1fr);
    gap: 9px;
    margin-bottom: 13px;
}

.validation-item {
    background: white;
    border: 1px solid #e2e8e4;
    border-radius: 10px;
    padding: 12px;
}

.validation-label {
    color: #7a8490;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
}

.validation-value {
    margin-top: 5px;
    font-size: 15px;
    font-weight: 780;
}

.validation-correct {
    color: #26715f;
}

.validation-missed {
    color: #9a3e24;
}

.test-performance {
    margin-top: 13px;
    padding: 13px 14px;
    background: white;
    border: 1px solid #e2e8e4;
    border-radius: 10px;
    color: #485568;
    font-size: 12px;
    line-height: 1.7;
}

.test-performance strong {
    color: #263349;
}

.validation-note {
    margin-top: 11px;
    color: #596577;
    font-size: 12px;
    line-height: 1.6;
}

.evidence-grid {
    display: grid;
    grid-template-columns:
        repeat(3, 1fr);
    gap: 9px;
}

.evidence-item {
    background: #f8fafc;
    border-radius: 9px;
    padding: 11px;
}

.evidence-label {
    color: #7c8798;
    font-size: 11px;
    margin-bottom: 4px;
}

.evidence-value {
    font-size: 14px;
    font-weight: 700;
}

.explainer-box {
    margin-top: 12px;
    padding: 13px 14px;
    background: #f7f9fc;
    border-radius: 10px;
    color: #596577;
    font-size: 12px;
    line-height: 1.6;
}

.explainer-box strong {
    color: #273349;
}

.reasoning-intro {
    font-size: 13px;
    color: #596577;
    line-height: 1.6;
    margin-bottom: 15px;
}

.rank-note {
    background: #f7f9fc;
    padding: 12px 14px;
    border-radius: 10px;
    margin-bottom: 12px;
    font-size: 12px;
    color: #596577;
    line-height: 1.6;
}

.driver {
    display: grid;

    grid-template-columns:
        48px
        185px
        95px
        1fr;

    gap: 10px;

    align-items: start;

    padding: 13px 0;

    border-top:
        1px solid #edf0f3;
}

.driver-rank {
    font-weight: 800;
    color: #4f5f78;
    font-size: 13px;
}

.driver-name {
    font-weight: 740;
    font-size: 13px;
}

.driver-impact {
    font-size: 13px;
    font-weight: 780;
}

.driver-description {
    font-size: 13px;
    color: #606c7e;
    line-height: 1.5;
}

.positive {
    color: #9a3e24;
}

.negative {
    color: #26715f;
}

.validation {
    margin-top: 14px;
    padding: 12px 14px;
    border-radius: 9px;
    background: #f3f7f5;
    font-size: 12px;
    line-height: 1.55;
    color: #40574f;
}

.small-grid {
    display: grid;
    grid-template-columns:
        repeat(2, 1fr);
    gap: 9px;
}

.small-item {
    background: #f8fafc;
    border-radius: 9px;
    padding: 11px;
    font-size: 13px;
}

.error {
    display: none;
    margin-top: 16px;
    padding: 13px;
    border-radius: 10px;
    background: #fff1f1;
    color: #8a2929;
}

.footer {
    margin-top: 17px;
    text-align: center;
    font-size: 11px;
    color: #8992a1;
}

@media (
    max-width: 700px
) {

    .metrics,
    .evidence-grid,
    .small-grid,
    .validation-grid {
        grid-template-columns:
            1fr 1fr;
    }

    .driver {
        grid-template-columns:
            1fr;
    }
}

</style>

</head>

<body>

<div class="page">

    <div class="header">

        <div class="brand">
            GridSentinel AI
        </div>

        <div class="subtitle">
            Governed AI for extreme-weather
            grid reliability and outage-risk
            assessment.
        </div>

        <div class="badge">
            Harris County · Texas
        </div>

    </div>


    <div class="card">

        <div class="field">

            <label>
                Assessment date
            </label>

            <input
                id="date"
                type="date"
                value="2024-07-08"
            >

        </div>


        <div class="field">

            <label>
                Ask GridSentinel
            </label>

            <textarea id="query">What happened during Hurricane Beryl and how severe was the outage?</textarea>

        </div>


        <button
            id="submit"
            onclick="assessRisk()"
        >
            Assess Grid Risk
        </button>


        <div
            id="error"
            class="error"
        ></div>


        <div
            id="result"
            class="result"
        >

            <div class="metrics">

                <div class="metric">
                    <div class="metric-name">
                        Risk
                    </div>
                    <div
                        id="risk"
                        class="metric-value"
                    ></div>
                </div>

                <div class="metric">
                    <div class="metric-name">
                        Probability
                    </div>
                    <div
                        id="probability"
                        class="metric-value"
                    ></div>
                </div>

                <div class="metric">
                    <div class="metric-name">
                        Human Review
                    </div>
                    <div
                        id="review"
                        class="metric-value"
                    ></div>
                </div>

                <div class="metric">
                    <div class="metric-name">
                        Latency
                    </div>
                    <div
                        id="latency"
                        class="metric-value"
                    ></div>
                </div>

            </div>


            <div class="section">

                <div class="section-title">
                    Risk Assessment
                </div>

                <div
                    id="summary"
                    class="summary"
                ></div>

            </div>


            <div
                class="section validation-highlight"
            >

                <div class="section-title">
                    Historical Validation
                </div>

                <div
                    id="historical-validation"
                ></div>

                <div
                    id="test-performance"
                    class="test-performance"
                ></div>

            </div>


            <div class="section">

                <div class="section-title">
                    Evidence Snapshot
                </div>

                <div
                    id="evidence"
                    class="evidence-grid"
                ></div>

                <div
                    id="evidence-help"
                    class="explainer-box"
                ></div>

            </div>


            <div class="section">

                <div class="section-title">
                    Why the Model Reached This Risk
                </div>

                <div
                    id="reasoning-intro"
                    class="reasoning-intro"
                ></div>

                <div class="rank-note">
                    <strong>How to read this table:</strong>
                    Rank #1 is the strongest influence
                    among the understandable factors
                    shown below for the selected date.
                    A positive score increased the
                    model's estimated outage risk.
                    A negative score reduced it.
                    The score is a Logistic Regression
                    model contribution in log-odds
                    units — it is <strong>not a
                    percentage</strong>. Factors used
                    internally by the model but not
                    useful for a clear client
                    explanation are intentionally not
                    displayed here.
                </div>

                <div id="drivers"></div>

                <div
                    id="validation"
                    class="validation"
                ></div>

            </div>


            <div class="section">

                <div class="section-title">
                    Governance & Grounding
                </div>

                <div class="small-grid">

                    <div
                        id="graph"
                        class="small-item"
                    ></div>

                    <div
                        id="runtime"
                        class="small-item"
                    ></div>

                </div>

            </div>

        </div>

    </div>


    <div class="footer">
        Decision-support prototype.
        Safety-critical actions require
        qualified human engineering review.
    </div>

</div>


<script>

function prettyDate(
    dateString
) {

    const date =
        new Date(
            dateString
            + "T00:00:00"
        );

    return date.toLocaleDateString(
        "en-US",
        {
            year: "numeric",
            month: "long",
            day: "numeric"
        }
    );
}


function addEvidence(
    container,
    label,
    value
) {

    if (
        value === undefined
        || value === null
        || value === ""
        || value === "NaN"
    ) {
        return;
    }

    const item =
        document.createElement(
            "div"
        );

    item.className =
        "evidence-item";

    item.innerHTML =
        "<div class='evidence-label'>"
        + label
        + "</div>"
        + "<div class='evidence-value'>"
        + value
        + "</div>";

    container.appendChild(
        item
    );
}


function driverExplanation(
    driver,
    selectedDate,
    rank
) {

    const dateText =
        prettyDate(
            selectedDate
        );

    const contribution =
        Math.abs(
            driver.contribution
        ).toFixed(2);

    const direction =
        driver.contribution > 0
        ? "increased"
        : "reduced";

    if (
        driver.feature
        === "storm_event"
    ) {

        return (
            "A severe-weather event was recorded "
            + "on the selected date ("
            + dateText
            + "). In this prediction, that signal "
            + direction
            + " the model's risk score by "
            + contribution
            + " units. It ranks #"
            + rank
            + " among the interpretable factors "
            + "shown here."
        );
    }


    if (
        driver.feature
        === "storm_recent_7d"
    ) {

        return (
            "The model also found severe weather "
            + "during the 7 days before "
            + dateText
            + ". That recent storm activity "
            + direction
            + " the risk score by "
            + contribution
            + " units. This means the model used "
            + "recent conditions as additional "
            + "context for the selected day."
        );
    }


    if (
        driver.feature
        === "outage_max_7d"
    ) {

        return (
            "This feature looks at the largest "
            + "customer outage observed during "
            + "the 7 days before "
            + dateText
            + ". In this prediction it "
            + direction
            + " the model's risk score by "
            + contribution
            + " units."
        );
    }


    if (
        driver.feature
        === "outage_mean_7d"
    ) {

        return (
            "This feature summarizes the average "
            + "customer outage level during the "
            + "7 days before "
            + dateText
            + ". For this specific prediction, "
            + "its value "
            + direction
            + " the model's risk score by "
            + contribution
            + " units. A negative contribution "
            + "does not mean outages are good; it "
            + "only describes how this trained "
            + "model used that value in this case."
        );
    }


    if (
        driver.feature
        === "outage_lag_1d"
    ) {

        return (
            "This feature is the customer outage "
            + "level one day before "
            + dateText
            + ". In this prediction it "
            + direction
            + " the model's risk score by "
            + contribution
            + " units. It provides the model with "
            + "very recent grid-condition context."
        );
    }


    if (
        driver.feature
        === "is_hurricane"
    ) {

        return (
            "The selected date was identified as "
            + "a hurricane event. In this "
            + "prediction, that indicator "
            + direction
            + " the model's risk score by "
            + contribution
            + " units."
        );
    }


    if (
        driver.feature
        === "is_derecho"
    ) {

        return (
            "The selected date was identified as "
            + "a derecho event. In this "
            + "prediction, that indicator "
            + direction
            + " the model's risk score by "
            + contribution
            + " units."
        );
    }


    if (
        driver.feature
        === "is_tornado"
    ) {

        return (
            "A tornado indicator was active for "
            + "the selected date. In this "
            + "prediction it "
            + direction
            + " the model's risk score by "
            + contribution
            + " units."
        );
    }


    if (
        driver.feature
        === "is_flood"
    ) {

        return (
            "A flood indicator was active for "
            + "the selected date. In this "
            + "prediction it "
            + direction
            + " the model's risk score by "
            + contribution
            + " units."
        );
    }


    return (
        driver.description
        + " In this prediction it "
        + direction
        + " the model's risk score by "
        + contribution
        + " units."
    );
}


function renderDriver(
    container,
    driver,
    rank,
    selectedDate
) {

    const row =
        document.createElement(
            "div"
        );

    row.className =
        "driver";

    const positive =
        driver.contribution > 0;

    const sign =
        positive
        ? "+"
        : "";

    const directionLabel =
        positive
        ? "Higher risk"
        : "Lower risk";

    row.innerHTML =
        "<div class='driver-rank'>#"
        + rank
        + "</div>"

        + "<div class='driver-name'>"
        + driver.label
        + "</div>"

        + "<div class='driver-impact "
        + (
            positive
            ? "positive"
            : "negative"
        )
        + "'>"
        + sign
        + driver.contribution.toFixed(2)
        + "<br>"
        + directionLabel
        + "</div>"

        + "<div class='driver-description'>"
        + driverExplanation(
            driver,
            selectedDate,
            rank
        )
        + "</div>";

    container.appendChild(
        row
    );
}


function renderHistoricalValidation(
    validation
) {

    const container =
        document.getElementById(
            "historical-validation"
        );

    if (!validation) {

        container.innerHTML =
            "<div class='validation-note'>"
            + "Historical validation is not available "
            + "for this date."
            + "</div>";

        return;
    }

    const statusText =
        validation.correct
        ? "CORRECT ✓"
        : "MISSED";

    const statusClass =
        validation.correct
        ? "validation-correct"
        : "validation-missed";

    const periodMessage =
        validation.is_out_of_sample
        ? (
            "This date belongs to the held-out "
            + "July-December 2024 test period and "
            + "was not used during model training."
        )
        : (
            "This date belongs to the training "
            + "period, so this comparison should "
            + "not be interpreted as an "
            + "out-of-sample validation result."
        );

    container.innerHTML =
        "<div class='validation-grid'>"

        + "<div class='validation-item'>"
        + "<div class='validation-label'>"
        + "Predicted"
        + "</div>"
        + "<div class='validation-value'>"
        + validation.predicted_label
        + "</div>"
        + "</div>"

        + "<div class='validation-item'>"
        + "<div class='validation-label'>"
        + "Actual Outcome"
        + "</div>"
        + "<div class='validation-value'>"
        + validation.actual_label
        + "</div>"
        + "</div>"

        + "<div class='validation-item'>"
        + "<div class='validation-label'>"
        + "Validation"
        + "</div>"
        + "<div class='validation-value "
        + statusClass
        + "'>"
        + statusText
        + "</div>"
        + "</div>"

        + "</div>"

        + "<div class='validation-note'>"
        + "<strong>Observed peak outage:</strong> "
        + validation
            .actual_peak_customers_out
            .toLocaleString()
        + " customers.<br>"
        + "<strong>"
        + validation.period
        + ":</strong> "
        + periodMessage
        + "</div>";
}


function renderTestPerformance(
    perf
) {

    const container =
        document.getElementById(
            "test-performance"
        );

    container.innerHTML =
        "<strong>"
        + "Held-Out Test Performance "
        + "(Jul-Dec 2024)"
        + "</strong><br>"

        + "ROC-AUC: "
        + perf.roc_auc.toFixed(3)

        + " · PR-AUC: "
        + perf.pr_auc.toFixed(3)

        + " · Precision: "
        + (
            perf.precision
            * 100
        ).toFixed(1)
        + "%"

        + " · Recall: "
        + (
            perf.recall
            * 100
        ).toFixed(1)
        + "%"

        + " · F1: "
        + perf.f1.toFixed(2)

        + "<br>"

        + "<strong>Major-outage days detected: "
        + perf.detected_major_outages
        + " of "
        + perf.total_major_outages
        + "</strong>"

        + "<br>"

        + "These metrics were measured on the "
        + "held-out July-December 2024 period, "
        + "which was not used to train the model.";
}


async function assessRisk() {

    const button =
        document.getElementById(
            "submit"
        );

    const result =
        document.getElementById(
            "result"
        );

    const error =
        document.getElementById(
            "error"
        );

    const selectedDate =
        document.getElementById(
            "date"
        ).value;

    button.disabled = true;

    button.textContent =
        "Analyzing...";

    result.style.display =
        "none";

    error.style.display =
        "none";


    try {

        const response =
            await fetch(
                "/assess",
                {
                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            {
                                query:
                                    document
                                    .getElementById(
                                        "query"
                                    )
                                    .value,

                                target_date:
                                    selectedDate
                            }
                        )
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail
                || "Assessment failed."
            );
        }


        document.getElementById(
            "risk"
        ).textContent =
            data.risk_label;


        document.getElementById(
            "probability"
        ).textContent =
            (
                data.risk_probability
                * 100
            ).toFixed(1)
            + "%";


        document.getElementById(
            "review"
        ).textContent =
            data.human_review_required
            ? "Required"
            : "Not required";


        document.getElementById(
            "latency"
        ).textContent =
            data.latency_seconds
            .toFixed(2)
            + " s";


        document.getElementById(
            "summary"
        ).textContent =
            data.risk_summary;


        renderHistoricalValidation(
            data.historical_validation
        );


        renderTestPerformance(
            data.test_performance
        );


        const evidence =
            document.getElementById(
                "evidence"
            );

        evidence.innerHTML = "";


        addEvidence(
            evidence,
            "Date",
            data.evidence[
                "Date"
            ]
            || prettyDate(
                selectedDate
            )
        );


        addEvidence(
            evidence,
            "Location",
            data.evidence[
                "Location"
            ]
        );


        addEvidence(
            evidence,
            "Peak customers out",
            Number(
                data.evidence[
                    "Peak customers without power"
                ]
            ).toLocaleString()
        );


        addEvidence(
            evidence,
            "Mean customers out",
            Math.round(
                Number(
                    data.evidence[
                        "Mean customers without power"
                    ]
                )
            ).toLocaleString()
        );


        addEvidence(
            evidence,
            "Peak outage rate",
            (
                Number(
                    data.evidence[
                        "Peak outage rate"
                    ]
                )
                * 100
            ).toFixed(1)
            + "%"
        );


        addEvidence(
            evidence,
            "Storm type",
            data.evidence[
                "Storm types"
            ]
        );


        document.getElementById(
            "evidence-help"
        ).innerHTML =
            "<strong>What these numbers mean:</strong><br>"
            + "<strong>Peak customers out</strong> is "
            + "the highest observed number of tracked "
            + "customers without power at any observation "
            + "on the selected date. "
            + "<strong>Mean customers out</strong> is "
            + "the average observed number without power "
            + "across that day's outage observations. "
            + "<strong>Peak outage rate</strong> is the "
            + "highest observed share of tracked customers "
            + "without power on that date. "
            + "<strong>Storm type</strong> identifies the "
            + "weather event associated with the selected "
            + "date in the evidence data.";


        document.getElementById(
            "reasoning-intro"
        ).innerHTML =
            "The factors below explain how the deployed "
            + "Logistic Regression model reached its "
            + "prediction for <strong>"
            + prettyDate(
                selectedDate
            )
            + "</strong>. They are ordered by the "
            + "absolute size of their influence among "
            + "the interpretable factors shown here. "
            + "A factor near the top had a larger effect "
            + "on this specific prediction than a factor "
            + "lower in the table.";


        const drivers =
            document.getElementById(
                "drivers"
            );

        drivers.innerHTML = "";


        const allDrivers = [
            ...data.reasoning
                .positive_drivers,

            ...data.reasoning
                .negative_drivers
        ];


        allDrivers.sort(
            (
                a,
                b
            ) =>
                Math.abs(
                    b.contribution
                )
                -
                Math.abs(
                    a.contribution
                )
        );


        allDrivers.forEach(
            (
                driver,
                index
            ) => {

                renderDriver(
                    drivers,
                    driver,
                    index + 1,
                    selectedDate
                );
            }
        );


        const modelProbability =
            (
                data.reasoning
                .model_probability
                * 100
            ).toFixed(1);


        const reconstructed =
            (
                data.reasoning
                .reconstructed_probability
                * 100
            ).toFixed(1);


        document.getElementById(
            "validation"
        ).innerHTML =
            "<strong>Exact model check:</strong> "
            + "The production model predicted "
            + modelProbability
            + "% and the explanation reconstructed "
            + reconstructed
            + "%. The numerical difference was "
            + data.reasoning
                .reconstruction_error
                .toExponential(1)
            + ". This means the explanation is "
            + "mathematically tied to the deployed "
            + "Logistic Regression calculation rather "
            + "than being an explanation invented by "
            + "the language model.";


        document.getElementById(
            "graph"
        ).innerHTML =
            "<strong>Knowledge Graph</strong><br>"
            + data.graph_context;


        document.getElementById(
            "runtime"
        ).innerHTML =
            "<strong>Runtime</strong><br>"
            + data.llm_provider
            + " · "
            + data.llm_model
            + "<br>Fallback: "
            + (
                data.llm_fallback_used
                ? "Yes"
                : "No"
            );


        result.style.display =
            "block";

    }

    catch (err) {

        error.textContent =
            err.message;

        error.style.display =
            "block";
    }

    finally {

        button.disabled =
            false;

        button.textContent =
            "Assess Grid Risk";
    }
}

</script>

</body>
</html>
"""