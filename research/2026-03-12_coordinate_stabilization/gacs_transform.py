"""
GACS Coordinate System Transform (v1.0)
Standalone module for transforming new scene embeddings into GACS coordinates.

Usage:
    from gacs_transform import load_coordinate_system, transform_new_scenes

    cs = load_coordinate_system("research/gacs_coordinate_system_v1.pkl")
    results = transform_new_scenes(cs, new_embeddings_768d)
    # results["coordinates"]   -> (N, 14) array
    # results["cluster_labels"] -> (N,) array
    # results["axis_names"]    -> list of 14 axis names
"""

import pickle

import numpy as np


def load_coordinate_system(pkl_path: str) -> dict:
    """Load a saved GACS coordinate system from a pickle file."""
    with open(pkl_path, "rb") as f:
        cs = pickle.load(f)
    print(f"Loaded GACS coordinate system v{cs['metadata']['version']}")
    print(f"  Trained on {cs['metadata']['n_scenes_trained']} scenes")
    print(f"  Embedding model: {cs['metadata']['embedding_model']}")
    print(f"  Date: {cs['metadata']['date']}")
    return cs


def transform_new_scenes(cs: dict, embeddings_768d: np.ndarray) -> dict:
    """
    Transform new scene embeddings into GACS coordinates.

    Parameters
    ----------
    cs : dict
        Loaded coordinate system (from load_coordinate_system).
    embeddings_768d : np.ndarray
        Shape (N, 768) array of emotion-distilroberta embeddings.

    Returns
    -------
    dict with keys:
        coordinates : np.ndarray, shape (N, 14)
        cluster_labels : np.ndarray, shape (N,)
        axis_names : list of str
        axis_variance : list of float
    """
    if embeddings_768d.ndim == 1:
        embeddings_768d = embeddings_768d.reshape(1, -1)
    assert embeddings_768d.shape[1] == 768, (
        f"Expected 768-dim embeddings, got {embeddings_768d.shape[1]}"
    )

    # Apply saved scaler -> PCA
    X_scaled = cs["scaler"].transform(embeddings_768d)
    coordinates = cs["pca"].transform(X_scaled)

    # Predict cluster labels
    cluster_labels = cs["kmeans"].predict(coordinates)

    return {
        "coordinates": coordinates,
        "cluster_labels": cluster_labels,
        "axis_names": cs["axis_names"],
        "axis_variance": cs["axis_variance"],
    }


if __name__ == "__main__":
    import sys
    # Quick test
    cs = load_coordinate_system(
        sys.argv[1] if len(sys.argv) > 1
        else "research/gacs_coordinate_system_v1.pkl"
    )
    # Generate a random test embedding
    rng = np.random.RandomState(0)
    test_emb = rng.randn(3, 768).astype(np.float32)
    result = transform_new_scenes(cs, test_emb)
    print(f"\nTest: {test_emb.shape[0]} scenes -> "
          f"coordinates {result['coordinates'].shape}, "
          f"clusters {result['cluster_labels']}")
