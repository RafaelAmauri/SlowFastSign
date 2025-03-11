import numpy as np
import json
import os

from collections import defaultdict
from tqdm import tqdm


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

    norm_feat1 = np.linalg.norm(feature1, axis=1).reshape(-1, 1)
    norm_feat2 = np.linalg.norm(feature2, axis=1).reshape(1, -1)
    cosine     = numerator / (norm_feat1 * norm_feat2)

    maxPerFrame = np.max(cosine, axis=1)
    
    return np.mean(maxPerFrame).astype(float)


def rankSimiliratyByFeatures(labeledFeaturesPath, unlabeledFeaturesPath):
    featuresLabeledSet   = dict()
    featuresUnlabeledSet = dict()
    similarityRank       = defaultdict(lambda: 0)

    # Get the file paths for the features in the labeled set
    for file in os.listdir(labeledFeaturesPath):
        filePath    = os.path.join(labeledFeaturesPath, file)
        fileContent = np.load(filePath, allow_pickle=True)
        featuresLabeledSet[filePath] = fileContent.item()['features'].numpy()

    # Get the file paths for the features in the unlabeled set
    for file in os.listdir(unlabeledFeaturesPath):
        filePath    = os.path.join(unlabeledFeaturesPath, file)
        fileContent = np.load(filePath, allow_pickle=True)
        featuresUnlabeledSet[filePath] = (fileContent.item()['features'].numpy(), fileContent.item()['confidence'])


    # For every sample in the unlabeled set, compare it to every sample in the labeled set and save the highest similarity
    # value achieved by it in 'similarityRank'
    for nameUnlabeledFeature, (unlabeledFeature, unlabeledConfidence) in tqdm(featuresUnlabeledSet.items()):
        # Strip the string and the _features.npy suffix to get only the name of the video.
        # This will be something like "01April_2010_Thursday_heute_default-3"
        nameUnlabeledFeature = nameUnlabeledFeature.split("/")[-1].removesuffix("_features.npy")
        
        for _, labeledFeature in featuresLabeledSet.items():
            # The similarity is the confidence score + the cosine similarity. This aggregates uncertainty and
            # representativeness into a single score!
            similarity = (unlabeledConfidence + cosineSimilarity(unlabeledFeature, labeledFeature)) / 2
            if similarity > similarityRank[nameUnlabeledFeature]:
                similarityRank[nameUnlabeledFeature] = similarity

    similarityRank = dict(sorted(similarityRank.items(), key=lambda x:x[1]))

    with open("SimilarityRank.json", "w") as filePointer:
        json.dump(similarityRank, filePointer, indent=4)

    return similarityRank