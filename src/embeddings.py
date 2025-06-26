import os
from fastembed import ImageEmbedding, TextEmbedding, SparseTextEmbedding
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


image_model = ImageEmbedding(
    model_name="Qdrant/clip-ViT-B-32-vision", cache_dir="embedding_models"
)
dense_embedder = TextEmbedding(
    model_name="snowflake/snowflake-arctic-embed-m", cache_dir="embedding_models"
)
sparse_embedder = SparseTextEmbedding(
    model_name="Qdrant/minicoil-v1", cache_dir="embedding_models"
)


def generate_embeddings(img_path: str, text: str):
    """Generate embeddings for an image and its associated text.
    Args:
        img_path (str): Path to the image file.
        text (str): Associated text description for the image.
    Returns:
        dict: A dictionary containing the image path, text description,
              and their respective embeddings.
    """
    
    # Validate image path exists
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    # 1) Image embedding (note: model.embed expects a list of paths, so we wrap [img_path])
    logger.info(f"Processing {img_path}...")
    image_emb = list(image_model.embed([img_path]))[0]

    # 2) Dense text embedding
    dense_emb  = list(dense_embedder.embed(text))[0]

    # 3) Sparse text embedding
    sparse_emb = list(sparse_embedder.embed(text))[0]

    # Store them however you like; here we use a dict per item
    return({
        "image_path":        img_path,
        "description":       text,
        "image_embedding":   image_emb,
        "dense_embedding":   dense_emb,
        "sparse_embedding":  sparse_emb,
    })