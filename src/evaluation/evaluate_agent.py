import time
from datetime import datetime

import pandas as pd

from src.agents.grid_agent import agent


TEST_CASES = [
    {
        "name": "Beryl high-risk case",
        "query": (
            "What happened during Hurricane Beryl "
            "and how severe was the outage?"
        ),
        "target_date": "2024-07-08",
        "expected_risk_label": "HIGH RISK",
        "expected_evidence_date": "2024-07-08",
        "expected_human_review": True,
        "required_terms": [
            "Hurricane Beryl",
            "1660703",
            "Human engineering review required",
        ],
    },
    {
        "name": "Normal low-outage day",
        "query": (
            "What was the outage risk on July 4, 2024?"
        ),
        "target_date": "2024-07-04",
        "expected_risk_label": "LOW RISK",
        "expected_evidence_date": "2024-07-04",
        "expected_human_review": False,
        "required_terms": [
            "2024-07-04",
        ],
    },
]


results = []


for test in TEST_CASES:

    print("\n================================")
    print(test["name"])
    print("================================")

    start = time.perf_counter()

    try:

        result = agent.invoke(
            {
                "query": test["query"],
                "target_date": test["target_date"],
            }
        )

        latency = time.perf_counter() - start

        answer = result["final_answer"]

        checks = {
            "risk_label_match": (
                result["risk_label"]
                == test["expected_risk_label"]
            ),

            "evidence_date_match": (
                result["evidence_date"]
                == test["expected_evidence_date"]
            ),

            "human_review_match": (
                result["human_review_required"]
                == test["expected_human_review"]
            ),

            "required_terms_present": all(
                term in answer
                for term in test["required_terms"]
            ),
        }

        passed = all(
            checks.values()
        )

        print(
            "Risk label:",
            result["risk_label"]
        )

        print(
            "Risk probability:",
            round(
                result["risk_probability"],
                4
            )
        )

        print(
            "Evidence date:",
            result["evidence_date"]
        )

        print(
            "Human review:",
            result["human_review_required"]
        )

        print(
            "Latency seconds:",
            round(
                latency,
                2
            )
        )

        print(
            "Checks:",
            checks
        )

        print(
            "PASS:",
            passed
        )

        results.append(
            {
                "test_name":
                    test["name"],

                "target_date":
                    test["target_date"],

                "risk_label":
                    result["risk_label"],

                "risk_probability":
                    result["risk_probability"],

                "evidence_date":
                    result["evidence_date"],

                "human_review":
                    result[
                        "human_review_required"
                    ],

                "latency_seconds":
                    latency,

                "passed":
                    passed,

                "timestamp":
                    datetime.now().isoformat(),
            }
        )

    except Exception as exc:

        latency = time.perf_counter() - start

        print(
            "ERROR:",
            str(exc)
        )

        results.append(
            {
                "test_name":
                    test["name"],

                "target_date":
                    test["target_date"],

                "risk_label":
                    "ERROR",

                "risk_probability":
                    None,

                "evidence_date":
                    None,

                "human_review":
                    None,

                "latency_seconds":
                    latency,

                "passed":
                    False,

                "timestamp":
                    datetime.now().isoformat(),

                "error":
                    str(exc),
            }
        )


results_df = pd.DataFrame(
    results
)


print(
    "\n================================"
)

print(
    "LLMOps Evaluation Summary"
)

print(
    "================================"
)

print(
    results_df.to_string(
        index=False
    )
)


pass_rate = (
    results_df["passed"]
    .mean()
    * 100
)


print(
    "\nPass rate:"
)

print(
    f"{pass_rate:.1f}%"
)


print(
    "\nAverage latency:"
)

print(
    f"{results_df['latency_seconds'].mean():.2f} seconds"
)


OUTPUT_FILE = (
    "src/evaluation/"
    "agent_evaluation_results.csv"
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    "\nSaved evaluation results:"
)

print(
    OUTPUT_FILE
)