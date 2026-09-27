from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


DATA_FILE = Path("data/processed/harris_model_dataset_2024.csv")
MODEL_FILE = Path("src/models/artifacts/best_model.joblib")
META_FILE = Path("src/models/artifacts/model_metadata.joblib")
FIGURE_DIR = Path("docs/figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load data and trained model
# -----------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

model = joblib.load(MODEL_FILE)
metadata = joblib.load(META_FILE)

features = metadata["features"]

X = df[features].copy()


# -----------------------------
# Model predictions
# -----------------------------

pred_proba = model.predict_proba(X)[:, 1]
pred_label = (pred_proba >= 0.5).astype(int)

df["predicted_probability"] = pred_proba
df["predicted_label"] = pred_label


# -----------------------------
# Train surrogate tree
# -----------------------------

surrogate = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=10,
    random_state=42
)

surrogate.fit(X, pred_label)

rules = export_text(
    surrogate,
    feature_names=features
)

print("\nSurrogate tree rules:")
print(rules)


# -----------------------------
# Save tree figure
# -----------------------------

plt.figure(figsize=(14, 8))
plot_tree(
    surrogate,
    feature_names=features,
    class_names=["low_risk", "high_risk"],
    filled=True,
    rounded=True,
)
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "surrogate_tree_rules.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("\nSaved surrogate tree figure:")
print(FIGURE_DIR / "surrogate_tree_rules.png")


# -----------------------------
# Real case: Beryl day
# -----------------------------

case = df[
    df["date"] == pd.Timestamp("2024-07-08")
].copy()

if case.empty:
    raise ValueError("July 8, 2024 case not found.")

X_case = case[features].copy()

baseline_prob = model.predict_proba(X_case)[0, 1]

print("\nBeryl case date:")
print(case["date"].iloc[0])

print("\nBaseline probability:")
print(round(baseline_prob, 4))


# -----------------------------
# What-if scenario 1:
# remove storm signal
# -----------------------------

scenario_1 = X_case.copy()
scenario_1["storm_event"] = 0
scenario_1["storm_count"] = 0
scenario_1["storm_recent_7d"] = 0
scenario_1["storm_count_7d"] = 0

scenario_1_prob = model.predict_proba(scenario_1)[0, 1]

print("\nWhat-if scenario 1:")
print("No storm event and no recent storm signal")
print("Probability:", round(scenario_1_prob, 4))


# -----------------------------
# What-if scenario 2:
# same storm day, but lower recent outage burden
# -----------------------------

scenario_2 = X_case.copy()
scenario_2["outage_lag_1d"] = 10
scenario_2["outage_mean_7d"] = 10
scenario_2["outage_max_7d"] = 20

scenario_2_prob = model.predict_proba(scenario_2)[0, 1]

print("\nWhat-if scenario 2:")
print("Same storm signal, but lower prior outage burden")
print("Probability:", round(scenario_2_prob, 4))


# -----------------------------
# Summary table
# -----------------------------

summary = pd.DataFrame(
    {
        "scenario": [
            "baseline_beryl_day",
            "no_storm_signal",
            "lower_prior_outage_burden",
        ],
        "probability": [
            baseline_prob,
            scenario_1_prob,
            scenario_2_prob,
        ],
    }
)

print("\nCounterfactual summary:")
print(summary.to_string(index=False))