from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RAW_FILE = Path(
    "data/raw/eaglei_outages_2024.csv"
)

PROCESSED_DIR = Path(
    "data/processed"
)

FIGURE_DIR = Path(
    "docs/figures"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 1. Read EAGLE-I in chunks and keep Harris County only
# =========================================================

chunks = []

for chunk in pd.read_csv(
    RAW_FILE,
    chunksize=1_000_000
):

    state_clean = (
        chunk["state"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    county_clean = (
        chunk["county"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mask = (
        (state_clean == "texas")
        & (county_clean == "harris")
    )

    selected = chunk.loc[
        mask,
        [
            "fips_code",
            "county",
            "state",
            "customers_out",
            "run_start_time",
            "total_customers",
        ],
    ].copy()

    if not selected.empty:
        chunks.append(selected)


if not chunks:
    raise ValueError(
        "No Harris County, Texas rows were found."
    )


# =========================================================
# 2. Combine Harris County rows
# =========================================================

df = pd.concat(
    chunks,
    ignore_index=True
)


# =========================================================
# 3. Clean numeric fields
# =========================================================

df["customers_out"] = pd.to_numeric(
    df["customers_out"],
    errors="coerce"
)

df["total_customers"] = pd.to_numeric(
    df["total_customers"],
    errors="coerce"
)


# =========================================================
# 4. Parse timestamps
# =========================================================

df["run_start_time"] = pd.to_datetime(
    df["run_start_time"],
    errors="coerce",
)

df = df.dropna(
    subset=["run_start_time"]
)

df["date"] = (
    df["run_start_time"]
    .dt.date
)


# =========================================================
# 5. Data-quality validation
# =========================================================
# A valid outage rate requires:
# total_customers > 0
# customers_out >= 0
# customers_out <= total_customers
#
# We DO NOT modify the original outage counts.
# We only prevent invalid ratios from being used.
# =========================================================

df["invalid_customer_count"] = (
    df["total_customers"].isna()
    | df["customers_out"].isna()
    | (df["total_customers"] <= 0)
    | (df["customers_out"] < 0)
    | (
        df["customers_out"]
        > df["total_customers"]
    )
)


# =========================================================
# 6. Calculate outage rate only for valid rows
# =========================================================

df["outage_rate"] = np.nan

valid_mask = (
    ~df["invalid_customer_count"]
)

df.loc[
    valid_mask,
    "outage_rate"
] = (
    df.loc[
        valid_mask,
        "customers_out"
    ]
    /
    df.loc[
        valid_mask,
        "total_customers"
    ]
)


# =========================================================
# 7. Daily aggregation
# =========================================================

daily = (
    df.groupby(
        "date",
        as_index=False
    )
    .agg(
        daily_peak_outage=(
            "customers_out",
            "max"
        ),

        daily_mean_outage=(
            "customers_out",
            "mean"
        ),

        total_customers=(
            "total_customers",
            "median"
        ),

        daily_peak_outage_rate=(
            "outage_rate",
            "max"
        ),

        invalid_rate_rows=(
            "invalid_customer_count",
            "sum"
        ),
    )
)


# =========================================================
# 8. Save processed datasets
# =========================================================

df.to_csv(
    PROCESSED_DIR
    / "harris_outages_15min_2024.csv",
    index=False,
)

daily.to_csv(
    PROCESSED_DIR
    / "harris_outages_daily_2024.csv",
    index=False,
)


# =========================================================
# 9. QA output
# =========================================================

print(
    "\nHarris County 15-minute rows:"
)

print(
    len(df)
)


print(
    "\nDate range:"
)

print(
    df["run_start_time"].min()
)

print(
    df["run_start_time"].max()
)


print(
    "\nInvalid customer-count rows:"
)

print(
    int(
        df[
            "invalid_customer_count"
        ].sum()
    )
)


print(
    "\nDaily dataset shape:"
)

print(
    daily.shape
)


print(
    "\nTop 10 outage days:"
)

print(
    daily.sort_values(
        "daily_peak_outage",
        ascending=False
    ).head(10).to_string(
        index=False
    )
)


# =========================================================
# 10. Explicit Beryl QA check
# =========================================================

beryl_day = daily[
    daily["date"].astype(str)
    == "2024-07-08"
]

print(
    "\nBeryl day QA:"
)

print(
    beryl_day.to_string(
        index=False
    )
)


# =========================================================
# 11. Plot
# =========================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    pd.to_datetime(
        daily["date"]
    ),
    daily[
        "daily_peak_outage"
    ],
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Peak customers without power"
)

plt.title(
    "Harris County, Texas — "
    "Daily Peak Power Outages, 2024"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "harris_daily_outages_2024.png",
    dpi=150,
)

plt.close()


print(
    "\nSaved figure:"
)

print(
    "docs/figures/"
    "harris_daily_outages_2024.png"
)