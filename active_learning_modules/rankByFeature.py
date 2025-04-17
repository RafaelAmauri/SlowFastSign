import numpy as np
import json
import os

from collections import defaultdict
from tqdm import tqdm

from active_learning_modules.kcenter import kCenter, distance
from active_learning_modules.cosinesimilarity import cosineSimilarity


def readFeaturesFromFile(labeledFeaturesPath: str, unlabeledFeaturesPath: str)-> tuple:
    """
    This function reads the .npy files that store the features for the labeled and unlabeled sets 
    and returns them as a dict, where the keys are the names of the videos, and the values are the extracted features.

    Args:
        labeledFeaturesPath   (str): The path to the extracted labeled features
        unlabeledFeaturesPath (str): The path to the extracted unlabeled features
    """

    # The keys are the names of the videos, and the values are the extracted features
    featuresLabeledSet   = dict()
    featuresUnlabeledSet = dict()

    # Get the file paths for the features in the LABELED set
    for file in os.listdir(labeledFeaturesPath):
        filePath    = os.path.join(labeledFeaturesPath, file)
        fileContent = np.load(filePath, allow_pickle=True)
        
        # Get the features from the .npy file
        currentFeature = fileContent.item()['features'].numpy()

        # Strip the string and the _features.npy suffix to get only the name of the video.
        # The processed string will be something like "01April_2010_Thursday_heute_default-3"
        fileName = filePath.split("/")[-1].removesuffix("_features.npy")

        featuresLabeledSet[filePath] = currentFeature


    # Get the file paths for the features in the UNLABELED set.
    for file in os.listdir(unlabeledFeaturesPath):
        filePath    = os.path.join(unlabeledFeaturesPath, file)
        fileContent = np.load(filePath, allow_pickle=True)

        # Get the features from the .npy file
        currentFeature    = fileContent.item()['features'].numpy()

        # Strip the string and the _features.npy suffix to get only the name of the video.
        # The processed string will be something like "01April_2010_Thursday_heute_default-3"
        fileName = filePath.split("/")[-1].removesuffix("_features.npy")

        featuresUnlabeledSet[fileName] = currentFeature


    return featuresLabeledSet, featuresUnlabeledSet



def rankSimiliratyByFeatures(featuresLabeledSet: dict, featuresUnlabeledSet: dict, strategy: str, saveFolder: str) -> dict:
    """
    Given a set of features for the labeled and unlabeled sets, this function selects the top 'nLabelings' best samples
    in the unlabeled set that are the most different from the samples in the labeled set.

    Args:
        featuresLabeledSet   (dict): The dict containing the features for the labeled set.   Key = video name, Value = features
        featuresUnlabeledSet (dict): The dict containing the features for the unlabeled set. Key = video name, Value = features, confidence
        strategy              (str): The selection strategy. Can be "cosine" or "kcenter".
        saveFolder            (str): The folder where to save the rank for best samples

    Returns:
        dict: The ranking for every sample in the unlabeled set.
    """
    # For now, each sample has a default similarity score of "0"
    similarityRank       = defaultdict(lambda: 0)

    if strategy == "cosine":
        # For every sample in the unlabeled set, compare it to every sample in the labeled set and save the highest similarity
        # value achieved by it in 'similarityRank'
        for nameUnlabeledFeature, (unlabeledFeature, _) in tqdm(featuresUnlabeledSet.items()):
            # Strip the string and the _features.npy suffix to get only the name of the video.
            # The processed string will be something like "01April_2010_Thursday_heute_default-3"
            nameUnlabeledFeature = nameUnlabeledFeature.split("/")[-1].removesuffix("_features.npy")
            
            for _, labeledFeature in featuresLabeledSet.items():
                # The similarity is the confidence score + the cosine similarity. This aggregates uncertainty and
                # representativeness into a single score!
                similarity = cosineSimilarity(unlabeledFeature, labeledFeature) # + confidence

                if similarity < similarityRank[nameUnlabeledFeature]:
                    similarityRank[nameUnlabeledFeature] = similarity
        

        similarityRank = dict(sorted(similarityRank.items(), key=lambda x:x[1]))


    # TODO Ver qual métrica de distância é melhor
    #print(nameUnlabeledFeature, unlabeledFeature.shape)

    #a = np.expand_dims(unlabeledFeature[9], axis=0)
    #b = np.expand_dims(unlabeledFeature[26], axis=0)

    #print(f"Cosine Similarity:  {cosineSimilarity(a, b)}")
    #print(f"Eucledian Distance: {distance(a, b)}")

    #raise Exception

    elif strategy == "kcenter":
        similarityRank = kCenter(featuresUnlabeledSet, featuresLabeledSet)
        
        
    # Save the similarity ranking
    savePath = os.path.join(saveFolder, "SimilarityRank.json")
    with open(savePath, "w") as filePointer:
        json.dump(similarityRank, filePointer, indent=4)

    return similarityRank