from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DATA_FILE = Path(
    "data/processed/harris_model_dataset_2024.csv"
)

MODEL_FILE = Path(
    "src/models/artifacts/best_model.joblib"
)


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


CLIENT_FEATURES = {
    "storm_event": {
        "label": "Severe weather today",
        "description": (
            "A severe-weather event was recorded "
            "on the assessment day."
        ),
    },

    "storm_recent_7d": {
        "label": "Recent severe weather",
        "description": (
            "Severe weather was also observed "
            "during the previous 7 days."
        ),
    },

    "is_hurricane": {
        "label": "Hurricane event",
        "description": (
            "The assessment day was identified "
            "as a hurricane event."
        ),
    },

    "is_derecho": {
        "label": "Derecho event",
        "description": (
            "The assessment day was identified "
            "as a derecho event."
        ),
    },

    "is_tornado": {
        "label": "Tornado event",
        "description": (
            "A tornado indicator was active "
            "for this date."
        ),
    },

    "is_flood": {
        "label": "Flood event",
        "description": (
            "A flood indicator was active "
            "for this date."
        ),
    },

    "is_thunderstorm_wind": {
        "label": "Thunderstorm wind",
        "description": (
            "A thunderstorm-wind indicator "
            "was active for this date."
        ),
    },

    "outage_lag_1d": {
        "label": "Previous-day outage",
        "description": (
            "Customer outages recorded one day "
            "before the assessment."
        ),
    },

    "outage_mean_7d": {
        "label": "Recent average outage",
        "description": (
            "Average customer outages during "
            "the previous 7 days."
        ),
    },

    "outage_max_7d": {
        "label": "Recent maximum outage",
        "description": (
            "Largest customer outage observed "
            "during the previous 7 days."
        ),
    },
}


def sigmoid(value):

    return 1.0 / (
        1.0 + np.exp(-value)
    )


def explain_prediction(
    target_date: str,
):

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
            f"No data found for {target_date}"
        )

    X = row[FEATURES]

    imputer = model.named_steps[
        "imputer"
    ]

    scaler = model.named_steps[
        "scaler"
    ]

    logistic = model.named_steps[
        "model"
    ]

    X_imputed = imputer.transform(
        X
    )

    X_scaled = scaler.transform(
        X_imputed
    )

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

    final_log_odds = (
        intercept
        + contributions.sum()
    )

    reconstructed_probability = (
        sigmoid(
            final_log_odds
        )
    )

    model_probability = float(
        model.predict_proba(
            X
        )[0, 1]
    )

    reconstruction_error = abs(
        model_probability
        - reconstructed_probability
    )

    all_drivers = []

    for index, feature in enumerate(
        FEATURES
    ):

        if feature not in CLIENT_FEATURES:
            continue

        raw_value = float(
            X.iloc[0][feature]
        )

        contribution = float(
            contributions[index]
        )

        # Do not show inactive binary indicators.
        if (
            feature.startswith("is_")
            and raw_value == 0
        ):
            continue

        info = CLIENT_FEATURES[
            feature
        ]

        all_drivers.append(
            {
                "feature":
                    feature,

                "label":
                    info["label"],

                "description":
                    info["description"],

                "raw_value":
                    raw_value,

                "contribution":
                    contribution,

                "absolute_contribution":
                    abs(
                        contribution
                    ),
            }
        )

    all_drivers = sorted(
        all_drivers,
        key=lambda item:
            item[
                "absolute_contribution"
            ],
        reverse=True,
    )

    positive = [
        item
        for item in all_drivers
        if item["contribution"] > 0
    ][:3]

    negative = [
        item
        for item in all_drivers
        if item["contribution"] < 0
    ][:3]

    return {
        "model_probability":
            model_probability,

        "reconstructed_probability":
            float(
                reconstructed_probability
            ),

        "reconstruction_error":
            float(
                reconstruction_error
            ),

        "positive_drivers":
            positive,

        "negative_drivers":
            negative,
    }


if __name__ == "__main__":

    result = explain_prediction(
        "2024-07-08"
    )

    print(
        result
    )