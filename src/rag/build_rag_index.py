from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


DATA_FILE = Path(
    "data/processed/harris_model_dataset_2024.csv"
)

OUTPUT_DIR = Path(
    "src/rag/artifacts"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------
# Load processed project data
# ---------------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

df = df.sort_values(
    "date"
).reset_index(drop=True)


documents = []


# ---------------------------------
# Create one evidence document
# per day
# ---------------------------------

for _, row in df.iterrows():

    date = row["date"].strftime(
        "%Y-%m-%d"
    )

    text = f"""
Date: {date}

Location: Harris County, Texas.

Peak customers without power:
{int(row['daily_peak_outage'])}

Mean customers without power:
{round(row['daily_mean_outage'], 2)}

Peak outage rate:
{round(row['daily_peak_outage_rate'], 6)}

Storm event:
{int(row['storm_event'])}

Storm count:
{int(row['storm_count'])}

Storm types:
{row['storm_types']}

Hurricane indicator:
{int(row['is_hurricane'])}

Derecho indicator:
{int(row['is_derecho'])}

Tornado indicator:
{int(row['is_tornado'])}

Flood indicator:
{int(row['is_flood'])}

Thunderstorm wind indicator:
{int(row['is_thunderstorm_wind'])}

Storm observed in previous 7 days:
{int(row['storm_recent_7d'])}

Peak outage one day earlier:
{round(row['outage_lag_1d'], 2)}

Mean outage during previous 7 days:
{round(row['outage_mean_7d'], 2)}

Maximum outage during previous 7 days:
{round(row['outage_max_7d'], 2)}
""".strip()

    documents.append(
        {
            "date": date,
            "text": text,
            "source": "GridSentinel processed Harris County dataset",
        }
    )


print(
    "\nDocuments created:",
    len(documents)
)


# ---------------------------------
# Embedding model
# ---------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


texts = [
    doc["text"]
    for doc in documents
]


print(
    "\nGenerating embeddings..."
)


embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True,
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


print(
    "Embedding shape:",
    embeddings.shape
)


# ---------------------------------
# Build FAISS index
# ---------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)


print(
    "\nFAISS vectors:",
    index.ntotal
)


# ---------------------------------
# Save index + metadata
# ---------------------------------

faiss.write_index(
    index,
    str(
        OUTPUT_DIR
        / "grid_knowledge.faiss"
    )
)

joblib.dump(
    documents,
    OUTPUT_DIR
    / "knowledge_documents.joblib"
)


print(
    "\nSaved FAISS index:"
)

print(
    OUTPUT_DIR
    / "grid_knowledge.faiss"
)

print(
    "\nSaved document metadata:"
)

print(
    OUTPUT_DIR
    / "knowledge_documents.joblib"
)