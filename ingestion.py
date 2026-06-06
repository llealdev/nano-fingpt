import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

load_dotenv()

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION_NAME = "financial"
FILE_PATH = "./AAPL_10-K_1A_temp.md"

qdrant = QdrantClient(
    api_key=os.getenv("API_KEY_QDRANT"),
    url=os.getenv("QDRANT_URL"),
)

qdrant.delete_collection(collection_name=COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
)

with open(FILE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

points = []
for chunk in chunks:
    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_embedding = list(sparse_model.passage_embed([chunk]))[0].as_object()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={"dense": dense_embedding, "sparse": sparse_embedding},
        payload={"text": chunk, "source": FILE_PATH},
    )

    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

query_text = "what are the main financial risks?"
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        {"query": query_dense, "using": "dense", "limit": 10},
        {"query": query_sparse, "using": "sparse", "limit": 10},
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=3,
)

for r in results.points:
    print(f"Score {r.score}")
    print(f"Texto {r.payload['text'][:100]}...")
    print("--" * 80)
