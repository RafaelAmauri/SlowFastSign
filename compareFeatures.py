import numpy as np
import matplotlib.pyplot as plt
import random

import numpy as np
import json
import os

from collections import defaultdict
from tqdm import tqdm

from active_learning_modules.kcenter import kCenter


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


def rankSimiliratyByFeatures(featuresLabeledSet, featuresUnlabeledSet, strategy, nLabelings, medianConf):
    similarityRank       = defaultdict(lambda: 0)

    if strategy == "cosine":

        # For every sample in the unlabeled set, compare it to every sample in the labeled set and save the highest similarity
        # value achieved by it in 'similarityRank'
        for nameUnlabeledFeature, (unlabeledFeature, unlabeledConfidence) in tqdm(featuresUnlabeledSet.items()):

            for labeledFeature in featuresLabeledSet:
                # The similarity is the confidence score + the cosine similarity. This aggregates uncertainty and
                # representativeness into a single score!

                similarity = ( 1 - unlabeledConfidence + cosineSimilarity(unlabeledFeature, labeledFeature)) / 2
                if similarity > similarityRank[nameUnlabeledFeature]:
                    similarityRank[nameUnlabeledFeature] = similarity
        
        similarityRank = dict(sorted(similarityRank.items(), key=lambda x:x[1], reverse=True))

        return similarityRank
    
    elif strategy == "kcenter":
        newFeatUnlabeledSet = dict()
        for idx, (feature, conf) in featuresUnlabeledSet.items():
            if conf < medianConf:
                newFeatUnlabeledSet[idx] = feature

        return kCenter(newFeatUnlabeledSet, nLabelings)



xAxisLimit = 400
yAxisLimit = 400

nLabeledSamples    = 5
nUnlabeledSamples  = 1500
nFeatureDimensions = 2
nLabelings         = 150
strategy           = "kcenter" # Can be "cosine" or "kcenter"


unlabeledConfidences = np.asarray([ random.random() for i in range(nUnlabeledSamples)])
medianConf           = np.median(unlabeledConfidences)

labeledFeatures   = np.asarray([[random.randint(1, xAxisLimit) for _ in range(nFeatureDimensions)] for _ in range(nLabeledSamples)])
unlabeledFeatures = np.asarray([[random.randint(1, xAxisLimit) for _ in range(nFeatureDimensions)] for _ in range(nUnlabeledSamples)])

featuresUnlabeledSet = { i : [unlabeledFeatures[i], unlabeledConfidences[i]] for i in range(nUnlabeledSamples) }

fig, axs = plt.subplots(2, 2, figsize=(12, 6))

similarity           = rankSimiliratyByFeatures(labeledFeatures, 
                                                featuresUnlabeledSet,
                                                strategy,
                                                nLabelings,
                                                medianConf)

sc1 = axs[0][0].scatter(unlabeledFeatures[..., 0], unlabeledFeatures[..., 1], c=unlabeledConfidences, vmin=0, vmax=1, cmap='cool', s=100, marker='s')
axs[0][0].set_title('Unlabeled Set')
axs[0][0].set_xlabel('Feature 1')
axs[0][0].set_ylabel('Feature 2')
axs[0][0].set_xlim(0, xAxisLimit)
axs[0][0].set_ylim(0, yAxisLimit)
cbar1 = plt.colorbar(sc1, ax=axs[0][0])
cbar1.set_label('Confidence')


sc2 = axs[0][1].scatter(labeledFeatures[..., 0], labeledFeatures[..., 1], s=100, marker='s')
axs[0][1].set_title('Original Labeled Set')
axs[0][1].set_xlabel('Feature 1')
axs[0][1].set_ylabel('Feature 2')
axs[0][1].set_xlim(0, xAxisLimit)
axs[0][1].set_ylim(0, yAxisLimit)

addedSamples = 0
for idx, conf in similarity.items():
    if addedSamples < nLabelings:
        x    = featuresUnlabeledSet[idx][0][0]
        y    = featuresUnlabeledSet[idx][0][1]
        conf = featuresUnlabeledSet[idx][1]

        
        xy = np.asarray([x, y])
        xy = np.expand_dims(xy, axis=0)
        labeledFeatures = np.concatenate([labeledFeatures, xy], axis=0)

        sc3 = axs[1][0].scatter(x, y, c=conf, vmin=0, vmax=1, cmap='cool', s=100, marker='s')
        addedSamples += 1


axs[1][0].set_title('Chosen Samples')
axs[1][0].set_xlabel('Feature 1')
axs[1][0].set_ylabel('Feature 2')
axs[1][0].set_xlim(0, xAxisLimit)
axs[1][0].set_ylim(0, yAxisLimit)
cbar3 = plt.colorbar(sc3, ax=axs[1][0])
cbar3.set_label('Confidence')


sc4 = axs[1][1].scatter(labeledFeatures[..., 0], labeledFeatures[..., 1], s=100, marker='s')
axs[1][1].set_title('New Labeled Set')
axs[1][1].set_xlabel('Feature 1')
axs[1][1].set_ylabel('Feature 2')
axs[1][1].set_xlim(0, xAxisLimit)
axs[1][1].set_ylim(0, yAxisLimit)

plt.tight_layout()
plt.show()