from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

DATA_FILE = Path(
    "data/processed/harris_model_dataset_2024.csv"
)

MODEL_FILE = Path(
    "src/models/artifacts/best_model.joblib"
)

OUTPUT_FILE = Path(
    "src/models/artifacts/"
    "beryl_exact_reasoning.json"
)


# =========================================================
# Features
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
# Utilities
# =========================================================

def sigmoid(value):
    return 1.0 / (
        1.0 + np.exp(-value)
    )


def explain_prediction(
    target_date: str,
):

    # -----------------------------------------------------
    # Load model and data
    # -----------------------------------------------------

    model = joblib.load(
        MODEL_FILE
    )

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    target = pd.Timestamp(
        target_date
    )

    row = df[
        df["date"] == target
    ]

    if row.empty:

        raise ValueError(
            f"No data found for "
            f"{target_date}"
        )

    X = row[FEATURES]


    # -----------------------------------------------------
    # Extract pipeline components
    # -----------------------------------------------------

    imputer = model.named_steps[
        "imputer"
    ]

    scaler = model.named_steps[
        "scaler"
    ]

    logistic = model.named_steps[
        "model"
    ]


    # -----------------------------------------------------
    # Apply exact preprocessing used by model
    # -----------------------------------------------------

    X_imputed = imputer.transform(
        X
    )

    X_scaled = scaler.transform(
        X_imputed
    )


    # -----------------------------------------------------
    # Exact Logistic Regression decomposition
    # -----------------------------------------------------

    coefficients = (
        logistic.coef_[0]
    )

    intercept = float(
        logistic.intercept_[0]
    )

    contributions = (
        X_scaled[0]
        * coefficients
    )


    model_log_odds = (
        intercept
        + contributions.sum()
    )

    reconstructed_probability = (
        sigmoid(
            model_log_odds
        )
    )

    original_probability = float(
        model.predict_proba(
            X
        )[0, 1]
    )


    reconstruction_error = abs(
        original_probability
        - reconstructed_probability
    )


    # -----------------------------------------------------
    # Build feature explanation table
    # -----------------------------------------------------

    rows = []

    for index, feature in enumerate(
        FEATURES
    ):

        rows.append(
            {
                "feature":
                    feature,

                "raw_value":
                    float(
                        X.iloc[0][
                            feature
                        ]
                    ),

                "standardized_value":
                    float(
                        X_scaled[0][
                            index
                        ]
                    ),

                "coefficient":
                    float(
                        coefficients[
                            index
                        ]
                    ),

                "log_odds_contribution":
                    float(
                        contributions[
                            index
                        ]
                    ),
            }
        )


    explanation_df = pd.DataFrame(
        rows
    )


    explanation_df[
        "absolute_contribution"
    ] = explanation_df[
        "log_odds_contribution"
    ].abs()


    explanation_df = (
        explanation_df.sort_values(
            "absolute_contribution",
            ascending=False,
        )
    )


    # -----------------------------------------------------
    # Positive and negative drivers
    # -----------------------------------------------------

    positive = (
        explanation_df[
            explanation_df[
                "log_odds_contribution"
            ] > 0
        ]
        .head(5)
    )


    negative = (
        explanation_df[
            explanation_df[
                "log_odds_contribution"
            ] < 0
        ]
        .head(5)
    )


    # -----------------------------------------------------
    # Human-readable reasoning
    # -----------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        "GridSentinel Exact Model Reasoning"
    )

    print(
        "===================================="
    )

    print(
        f"\nDate: {target_date}"
    )

    print(
        f"Model probability: "
        f"{original_probability:.6f}"
    )

    print(
        f"Reconstructed probability: "
        f"{reconstructed_probability:.6f}"
    )

    print(
        f"Reconstruction error: "
        f"{reconstruction_error:.12f}"
    )

    print(
        f"\nIntercept / baseline log-odds: "
        f"{intercept:.4f}"
    )

    print(
        f"Final log-odds: "
        f"{model_log_odds:.4f}"
    )


    print(
        "\nTop risk-increasing drivers:"
    )

    if positive.empty:

        print(
            "- None"
        )

    else:

        for _, item in positive.iterrows():

            print(
                f"- {item['feature']}: "
                f"value="
                f"{item['raw_value']:.3f}, "
                f"contribution="
                f"+{item['log_odds_contribution']:.3f}"
            )


    print(
        "\nTop risk-reducing drivers:"
    )

    if negative.empty:

        print(
            "- None"
        )

    else:

        for _, item in negative.iterrows():

            print(
                f"- {item['feature']}: "
                f"value="
                f"{item['raw_value']:.3f}, "
                f"contribution="
                f"{item['log_odds_contribution']:.3f}"
            )


    # -----------------------------------------------------
    # Save machine-readable explanation
    # -----------------------------------------------------

    result = {
        "target_date":
            target_date,

        "model_probability":
            original_probability,

        "reconstructed_probability":
            float(
                reconstructed_probability
            ),

        "reconstruction_error":
            float(
                reconstruction_error
            ),

        "intercept":
            intercept,

        "final_log_odds":
            float(
                model_log_odds
            ),

        "top_positive_drivers":
            positive[
                [
                    "feature",
                    "raw_value",
                    "coefficient",
                    "log_odds_contribution",
                ]
            ].to_dict(
                orient="records"
            ),

        "top_negative_drivers":
            negative[
                [
                    "feature",
                    "raw_value",
                    "coefficient",
                    "log_odds_contribution",
                ]
            ].to_dict(
                orient="records"
            ),
    }


    OUTPUT_FILE.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "\nSaved explanation:"
    )

    print(
        OUTPUT_FILE
    )


    return result


# =========================================================
# Beryl demo
# =========================================================

if __name__ == "__main__":

    explain_prediction(
        "2024-07-08"
    )