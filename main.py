# %%
# This file contains functions to search for documents in a Qdrant collection
# based on text or image queries. It uses the FastEmbed library for embeddings
# and the Qdrant client for querying the vector database.

import os
from qdrant_client import QdrantClient, models
from fastembed import ImageEmbedding, TextEmbedding, SparseTextEmbedding


#%%
# Function to search for documents based on a text query ONLY
def search_by_text(
    query_text : str,
    collection_name: str = "document_embeddings",
    count: int = 100,
):
    # initialize client
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    # initialize embedder models
    dense_embedder = TextEmbedding(
        model_name="snowflake/snowflake-arctic-embed-m", cache_dir="embedding_models"
    )
    sparse_embedder = SparseTextEmbedding(
        model_name="Qdrant/minicoil-v1", cache_dir="embedding_models"
    )

    dense_vector = list(dense_embedder.embed(query_text))[0]
    sparse_vector = list(sparse_embedder.embed(query_text))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_vector.indices,
        values=sparse_vector.values,
    )

    fast_dense_prefetcher = models.Prefetch(
        query=dense_vector,
        using="snowflake_m",
        limit=1000,
    )

    prefetcher = models.Prefetch(
        prefetch=fast_dense_prefetcher, query=sparse_vector, using="minicoil", limit=25
    )

    response = client.query_points(
        collection_name=collection_name,
        prefetch=prefetcher,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit= count,
    )

    result = [i.payload for i in response.points]
    return result


#%%
# Function to search for documents based on an image query and optional text
# If no text is provided, it only uses the image embedding for the search.
def search_by_image(
    query_image: str,
    query_text: str = None,
    collection_name: str = "document_embeddings",
    count: int = 100,
):
    # Check if the image file exists
    if not os.path.isfile(query_image):
        raise FileNotFoundError(f"Image file not found: {query_image}")
    
    # Initialize client
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    image_model = ImageEmbedding(
            model_name="Qdrant/clip-ViT-B-32-vision", cache_dir="embedding_models"
        )
    image_emb = list(image_model.embed([query_image]))[0]

    # create prefetcher based on whether text is provided
    # if no text is provided, we only use the image embedding
    if query_text is None:     
        prefetcher = models.Prefetch(
            query=image_emb,
            using="ViT",
            limit=100,
        )
    else:
        dense_embedder = TextEmbedding(
            model_name="snowflake/snowflake-arctic-embed-m", cache_dir="embedding_models"
        )
        sparse_embedder = SparseTextEmbedding(
            model_name="Qdrant/minicoil-v1", cache_dir="embedding_models"
        )
        dense_vector = list(dense_embedder.embed(query_text))[0]
        sparse_vector = list(sparse_embedder.embed(query_text))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_vector.indices,
            values=sparse_vector.values,
        )
        image_prefetcher = models.Prefetch(
            query=image_emb,
            using="ViT",
            limit=100,
        )
        fast_dense_prefetcher = models.Prefetch(
            prefetch=image_prefetcher,
            query=dense_vector,
            using="snowflake_m",
            limit=50,
        )
        prefetcher = models.Prefetch(
            prefetch=fast_dense_prefetcher, 
            query=sparse_vector, 
            using="minicoil", limit=25
        )

    response = client.query_points(
        collection_name=collection_name,
        prefetch=prefetcher,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit= count,
    )

    result = [i.payload for i in response.points]
    return result
# %%
