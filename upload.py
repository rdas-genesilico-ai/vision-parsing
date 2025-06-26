#%%
import os
import random
from src.image2text import generate_text
from src.embeddings import generate_embeddings
import uuid

# vector db of your choice
from qdrant_client import QdrantClient, models

# Configuration
DATA_DIR = "data"
USE_SUBSET = False   # True -> random subset, False -> all images
SUBSET_SIZE = 5     # if USE_SUBSET is True

#%%
def get_image_paths(data_dir: str):
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    return [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.lower().endswith(valid_exts)
    ]

#%%
# Collect image paths from the specified directory
# Try to get texts for all the images in the directory
image_paths = get_image_paths(DATA_DIR)
if USE_SUBSET:
    image_paths = random.sample(image_paths, min(SUBSET_SIZE, len(image_paths)))

results = []
for img_path in image_paths:
    try:
        print(f"Processing image: {img_path}")
        text = generate_text(img_path)
        results.append((img_path, text))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# %%
# Generate embeddings for the collected image-text pairs
embeddings = []
for img_path, text in results:
    embeddings.append(generate_embeddings(img_path, text))

#%%
# Initialize Qdrant client
# You can use a vector database of your choice, here we use Qdrant
# Make sure Qdrant is running on localhost:6333 or adjust the URL accordingly
qdrant_client = QdrantClient(url="http://localhost:6333")
coll_name = "document_embeddings"

qdrant_client.delete_collection(coll_name)
qdrant_client.create_collection(
    collection_name=coll_name,
    vectors_config={
        "snowflake_m": models.VectorParams(size=768, distance=models.Distance.COSINE),
        "ViT": models.VectorParams(size=512, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "minicoil": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    },
)

#%%
# Prepare points for Qdrant
# Each point contains a unique ID, vector embeddings, and payload with image path and description
points = []
for item in embeddings:
    point_obj = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "snowflake_m": item["dense_embedding"],
            "ViT": item["image_embedding"],
            "minicoil": models.SparseVector(
                indices=item["sparse_embedding"].indices,
                values=item["sparse_embedding"].values,
            ),
        },
        payload={
            "image_path": item["image_path"],
            "description": item["description"],
        },
    )
    points.append(point_obj)
    

# %%
# upsert points to Qdrant collection
# Note: `wait=True` ensures the operation completes before proceeding
# may take a while depending on the number of points and where the Qdrant server is hosted
qdrant_client.upsert(
    collection_name=coll_name,
    points=points,
    wait=True,
)

# %%
