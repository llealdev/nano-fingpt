import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from utils.semantic_chunker import SemanticChunker
from utils.edgar_clinet import EdgarClient

load_dotenv()

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLLECTION_NAME = "financial"
EMAIL = "lleal.dev@gmail.com"
MAX_TOKENS = 300

qdrant = QdrantClient(
    api_key=os.getenv("API_KEY_QDRANT"),
    url=os.getenv("QDRANT_URL"),
)

edgar = EdgarClient(email=EMAIL)
data_10k = edgar.fetch_filing_data("AAPL", "10-K")
text_10k = edgar.get_combined_text(data_10k)

data_10q = edgar.fetch_filing_data("AAPL", "10-Q")
text_10q = edgar.get_combined_text(data_10q)

chunker = SemanticChunker(max_tokens=MAX_TOKENS)

all_chunks = []
for data, text in [(data_10k, text_10k), (data_10q, text_10q)]:
    chunks = chunker.create_chunks(text)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "metadata": data["metadata"]})

dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)

points = []
for chunk_data in all_chunks:
    chunk = chunk_data["text"]
    metadata = chunk_data["metadata"]

    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_embedding = list(sparse_model.passage_embed([chunk]))[0].as_object()
    colbert_embedding = list(colbert_model.passage_embed([chunk]))[0].tolist()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_embedding,
            "sparse": sparse_embedding,
            "colbert": colbert_embedding,
        },
        payload={"text": chunk, "source": metadata},
    )

    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points, batch_size=5)
