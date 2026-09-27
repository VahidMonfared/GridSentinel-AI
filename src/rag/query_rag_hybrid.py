from pathlib import Path

import faiss
import joblib
import numpy as np

from sentence_transformers import SentenceTransformer


ARTIFACT_DIR = Path(
    "src/rag/artifacts"
)

INDEX_FILE = (
    ARTIFACT_DIR
    / "grid_knowledge.faiss"
)

DOCS_FILE = (
    ARTIFACT_DIR
    / "knowledge_documents.joblib"
)


index = faiss.read_index(
    str(INDEX_FILE)
)

documents = joblib.load(
    DOCS_FILE
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


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
    dtype="float32",
)


# Retrieve more candidates first
scores, indices = index.search(
    query_embedding,
    20,
)


candidates = []


for score, idx in zip(
    scores[0],
    indices[0]
):

    doc = documents[idx]

    text_lower = doc["text"].lower()
    query_lower = query.lower()

    bonus = 0.0

    if (
        "hurricane" in query_lower
        and "hurricane indicator:\n1"
        in text_lower
    ):
        bonus += 0.30

    if (
        "beryl" in query_lower
        and "beryl" in text_lower
    ):
        bonus += 0.40

    if (
        "tornado" in query_lower
        and "tornado indicator:\n1"
        in text_lower
    ):
        bonus += 0.30

    if (
        "flood" in query_lower
        and "flood indicator:\n1"
        in text_lower
    ):
        bonus += 0.30

    final_score = float(
        score + bonus
    )

    candidates.append(
        {
            "score": final_score,
            "semantic_score": float(score),
            "doc": doc,
        }
    )


candidates = sorted(
    candidates,
    key=lambda x: x["score"],
    reverse=True,
)


print("\nQuery:")
print(query)

print(
    "\nTop hybrid retrieved evidence:"
)


for rank, item in enumerate(
    candidates[:5],
    start=1
):

    doc = item["doc"]

    print(
        "\n----------------------"
    )

    print(
        f"Rank: {rank}"
    )

    print(
        "Semantic score:",
        round(
            item["semantic_score"],
            4
        ),
    )

    print(
        "Hybrid score:",
        round(
            item["score"],
            4
        ),
    )

    print(
        "Date:",
        doc["date"]
    )

    print(
        doc["text"]
    )