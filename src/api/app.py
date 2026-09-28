import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.agents.grid_agent import agent
from src.evaluation.observability import log_agent_run


app = FastAPI(
    title="GridSentinel AI",
    version="1.0.0",
    description=(
        "Governed agentic AI for extreme-weather "
        "grid outage risk assessment."
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

    start_time = time.perf_counter()

    try:

        result = agent.invoke(
            {
                "query": request.query,
                "target_date": request.target_date,
            }
        )

        latency_seconds = (
            time.perf_counter()
            - start_time
        )

        log_agent_run(
            query=request.query,
            target_date=request.target_date,

            risk_probability=
                result["risk_probability"],

            risk_label=
                result["risk_label"],

            evidence_date=
                result["evidence_date"],

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

        return {
            "target_date":
                request.target_date,

            "risk_probability":
                result["risk_probability"],

            "risk_label":
                result["risk_label"],

            "evidence_date":
                result["evidence_date"],

            "human_review_required":
                result[
                    "human_review_required"
                ],

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

            "answer":
                result["final_answer"],
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get("/", response_class=HTMLResponse)
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

    <title>GridSentinel AI</title>

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

            background:
                #f7f8fa;

            color:
                #172033;
        }

        .page {
            max-width: 920px;
            margin: 0 auto;
            padding: 48px 22px;
        }

        .header {
            margin-bottom: 28px;
        }

        .brand {
            font-size: 30px;
            font-weight: 750;
            letter-spacing: -0.8px;
        }

        .subtitle {
            margin-top: 8px;
            color: #657085;
            font-size: 15px;
            line-height: 1.6;
        }

        .badge {
            display: inline-block;
            margin-top: 14px;
            padding: 6px 10px;
            border-radius: 20px;
            background: #e9eef8;
            color: #344765;
            font-size: 12px;
            font-weight: 600;
        }

        .card {
            background: white;
            border: 1px solid #e3e6eb;
            border-radius: 16px;
            padding: 24px;

            box-shadow:
                0 5px 20px
                rgba(0,0,0,0.04);
        }

        label {
            display: block;
            margin-bottom: 7px;
            font-size: 13px;
            font-weight: 650;
        }

        input,
        textarea {
            width: 100%;
            border: 1px solid #ccd2dc;
            border-radius: 10px;
            font: inherit;
            padding: 12px 14px;
            outline: none;
            background: white;
        }

        input:focus,
        textarea:focus {
            border-color: #63789c;

            box-shadow:
                0 0 0 3px
                rgba(99,120,156,0.12);
        }

        textarea {
            min-height: 120px;
            resize: vertical;
        }

        .field {
            margin-bottom: 18px;
        }

        button {
            width: 100%;
            border: 0;
            border-radius: 10px;
            padding: 13px 18px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            background: #172033;
            color: white;
        }

        button:hover {
            opacity: 0.92;
        }

        button:disabled {
            opacity: 0.55;
            cursor: wait;
        }

        .result {
            margin-top: 26px;
            display: none;
        }

        .metrics {
            display: grid;

            grid-template-columns:
                repeat(4, 1fr);

            gap: 12px;
            margin-bottom: 18px;
        }

        .metric {
            background: #f8f9fb;
            border: 1px solid #e4e7eb;
            border-radius: 11px;
            padding: 14px;
        }

        .metric-name {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            color: #7a8495;
        }

        .metric-value {
            margin-top: 5px;
            font-size: 17px;
            font-weight: 750;
        }

        .answer {
            white-space: pre-wrap;
            line-height: 1.65;
            padding: 18px;
            border-radius: 11px;
            background: #f8f9fb;
            border: 1px solid #e4e7eb;
            font-size: 14px;
        }

        .footer {
            margin-top: 18px;
            text-align: center;
            color: #8891a0;
            font-size: 11px;
        }

        .error {
            margin-top: 18px;
            display: none;
            padding: 14px;
            background: #fff1f1;
            border: 1px solid #efcaca;
            border-radius: 10px;
            color: #8a2929;
        }

        @media (
            max-width: 650px
        ) {
            .metrics {
                grid-template-columns: 1fr;
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
            Governed agentic AI for
            extreme-weather grid reliability
            and outage-risk assessment.
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

            <textarea
                id="query"
                placeholder="Ask about outage risk, severe weather, or historical grid conditions..."
            >What happened during Hurricane Beryl and how severe was the outage?</textarea>

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


            <div
                id="answer"
                class="answer"
            ></div>

        </div>

    </div>


    <div class="footer">

        Decision-support prototype.
        Safety-critical actions require
        qualified human review.

    </div>

</div>


<script>

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
                    method: "POST",

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
            "answer"
        ).textContent =
            data.answer;


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