import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.agents.grid_agent import agent
from src.evaluation.observability import log_agent_run
from src.models.reasoning_explainer import (
    explain_prediction,
)


app = FastAPI(
    title="GridSentinel AI",
    version="1.1.0",
    description=(
        "Governed agentic AI for extreme-weather "
        "grid outage risk assessment."
    ),
)


class AssessmentRequest(BaseModel):
    query: str
    target_date: str


def extract_risk_summary(
    final_answer: str,
):

    text = final_answer

    start_marker = (
        "1. Risk Assessment"
    )

    end_marker = (
        "2. Evidence"
    )

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

    start_time = (
        time.perf_counter()
    )

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

        latency_seconds = (
            time.perf_counter()
            - start_time
        )

        log_agent_run(
            query=
                request.query,

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
        rgba(20, 30, 50, 0.05);
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
    background: #ffffff;
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

.reasoning-intro {
    font-size: 13px;
    color: #667286;
    line-height: 1.55;
    margin-bottom: 13px;
}

.driver {
    display: grid;
    grid-template-columns:
        165px 80px 1fr;
    gap: 11px;

    align-items: center;

    padding: 10px 0;

    border-top:
        1px solid #edf0f3;
}

.driver:first-of-type {
    border-top: 0;
}

.driver-name {
    font-weight: 720;
    font-size: 13px;
}

.driver-impact {
    font-size: 13px;
    font-weight: 760;
}

.driver-description {
    font-size: 13px;
    color: #606c7e;
    line-height: 1.45;
}

.positive {
    color: #9a3e24;
}

.negative {
    color: #26715f;
}

.validation {
    margin-top: 13px;
    padding: 11px 13px;
    border-radius: 9px;
    background: #f3f7f5;
    font-size: 13px;
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
    .small-grid {
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


            <div class="section">

                <div class="section-title">
                    Evidence Snapshot
                </div>

                <div
                    id="evidence"
                    class="evidence-grid"
                ></div>

            </div>


            <div class="section">

                <div class="section-title">
                    Why the Model Reached This Risk
                </div>

                <div class="reasoning-intro">
                    These are direct contributions
                    from the deployed Logistic
                    Regression model. Positive values
                    push risk upward; negative values
                    push it downward. They describe
                    model influence, not causation.
                </div>

                <div
                    id="drivers"
                ></div>

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

function addEvidence(
    container,
    label,
    value
) {

    if (
        value === undefined
        || value === null
        || value === ""
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


function renderDriver(
    container,
    driver
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

    row.innerHTML =
        "<div class='driver-name'>"
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
        + "</div>"
        + "<div class='driver-description'>"
        + driver.description
        + "</div>";

    container.appendChild(
        row
    );
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
                                    document
                                    .getElementById(
                                        "date"
                                    )
                                    .value
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


        const drivers =
            document.getElementById(
                "drivers"
            );

        drivers.innerHTML = "";


        data.reasoning
        .positive_drivers
        .forEach(
            driver =>
                renderDriver(
                    drivers,
                    driver
                )
        );


        data.reasoning
        .negative_drivers
        .forEach(
            driver =>
                renderDriver(
                    drivers,
                    driver
                )
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
        ).textContent =
            "Exact validation: model "
            + modelProbability
            + "% · reconstructed "
            + reconstructed
            + "% · numerical error "
            + data.reasoning
                .reconstruction_error
                .toExponential(1);


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

