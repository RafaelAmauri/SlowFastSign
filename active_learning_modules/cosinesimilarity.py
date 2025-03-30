import numpy as np


def cosineSimilarity(feature1, feature2):
    """
    Calculates cosine similarity between two feature tensors.
    Args:
        feature1 (np.typing.ArrayLike): The first feature tensor.
        feature2 (np.typing.ArrayLike): The second feature tensor.

    Returns:
        float: The cosine similarity.
    """
    # Calculate cosine similarity
    numerator  = np.dot(feature1, feature2.T)

    norm_feat1 = np.linalg.norm(feature1, axis=0).reshape(-1, 1)
    norm_feat2 = np.linalg.norm(feature2, axis=0).reshape(1, -1)
    cosine     = numerator / (norm_feat1 * norm_feat2)
    
    maxPerFrame = np.max(cosine, axis=1)
    
    return np.mean(maxPerFrame).astype(float)