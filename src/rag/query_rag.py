from pathlib import Path

import faiss
import joblib
import numpy as np

from sentence_transformers import SentenceTransformer


ARTIFACT_DIR = Path(
    "src/rag/artifacts"
)

INDEX_FILE = ARTIFACT_DIR / "grid_knowledge.faiss"

DOCS_FILE = ARTIFACT_DIR / "knowledge_documents.joblib"


# -----------------------------
# Load FAISS index
# -----------------------------

index = faiss.read_index(
    str(INDEX_FILE)
)


# -----------------------------
# Load document metadata
# -----------------------------

documents = joblib.load(
    DOCS_FILE
)


# -----------------------------
# Load same embedding model
# -----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# Query
# -----------------------------

query = (
    "What happened during Hurricane Beryl "
    "and how severe was the outage?"
)


query_embedding = embedding_model.encode(
    [query],
    normalize_embeddings=True,
)

query_embedding = np.asarray(
    query_embedding,
    dtype="float32"
)


# -----------------------------
# Search top matches
# -----------------------------

TOP_K = 5

scores, indices = index.search(
    query_embedding,
    TOP_K,
)


print("\nQuery:")
print(query)

print("\nTop retrieved evidence:")


for rank, idx in enumerate(
    indices[0],
    start=1
):

    doc = documents[idx]

    score = scores[0][rank - 1]

    print(
        "\n----------------------"
    )

    print(
        f"Rank: {rank}"
    )

    print(
        f"Similarity score: "
        f"{score:.4f}"
    )

    print(
        f"Date: {doc['date']}"
    )

    print(
        doc["text"]
    )