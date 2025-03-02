import numpy as np
import os

#TODO Calcular a similaridade de features para todos e não para só um.
def rankSimiliratyByFeatures(labeledFeaturesPath, unlabeledFeaturesPath):
    # Each path is a folder containing the extracted features for each video.
    extractedFeatures = dict()

    for folder in [labeledFeaturesPath, unlabeledFeaturesPath]:
        for file in os.listdir(folder):
            filePath = os.path.join(folder, file)
            extractedFeatures[filePath] = np.load(filePath, allow_pickle=True).item()['features'].numpy()

    feat1 = extractedFeatures["/workspace/SlowFastSign/worktest/siamese-unlabeled-run-1-features-fodao/test/30May_2011_Monday_heute_default-18_features.npy"]
    feat2 = extractedFeatures["/workspace/SlowFastSign/worktest/siamese-unlabeled-run-1-features-fodao/test/15February_2011_Tuesday_heute_default-24_features.npy"]
    
    # Normalize the feature values
    feat1 = feat1 / feat1.max()
    feat2 = feat2 / feat2.max()
    print(feat1.shape, feat2.shape)

    feat1 = feat1.max(axis=0)
    feat2 = feat2.max(axis=0)

    feat1 = feat1.flatten()
    feat2 = feat2.flatten()
    

    # Calculate cosine similarity
    numerator = np.dot(feat1, feat2)
    denom     = np.linalg.norm(feat1) * np.linalg.norm(feat2)
    cosine    = numerator / denom
    
    print(cosine, cosine.shape)


labeledFeaturesPath   = "/workspace/SlowFastSign/worktest/siamese-labeled-run-1-features/train"
unlabeledFeaturesPath = "/workspace/SlowFastSign/worktest/siamese-unlabeled-run-1-features-fodao/test"

rankSimiliratyByFeatures(labeledFeaturesPath, unlabeledFeaturesPath)