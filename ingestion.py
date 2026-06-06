import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

load_dotenv()

MODEL_EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "financial"
FILE_PATH = "./AAPL_10-K_1A_temp.md"

qdrant = QdrantClient(
    api_key=os.getenv("API_KEY_QDRANT"),
    url=os.getenv("QDRANT_URL"),
)

qdrant.delete_collection(collection_name=COLLECTION_NAME)
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

with open(FILE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

model = TextEmbedding(model_name=MODEL_EMBED_NAME)

points = []
for chunk in chunks:
    embedding = list(model.passage_embed([chunk]))[0].tolist()
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"text": chunk, "source": FILE_PATH},
    )

    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

query_text = "what are the main financial risks?"
query_embedding = list(model.query_embed([query_text]))[0].tolist()

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=3,
)

for r in results.points:
    print(f"Score {r.score}")
    print(f"Texto {r.payload['text'][:100]}...")
    print("--" * 80)
