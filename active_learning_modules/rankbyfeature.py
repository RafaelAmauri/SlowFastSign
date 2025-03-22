import numpy as np
import json
import os

from collections import defaultdict
from tqdm import tqdm

from active_learning_modules.kcenter import kCenter
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

    confidences = []

    # Get the file paths for the features in the labeled set
    for file in os.listdir(labeledFeaturesPath):
        filePath    = os.path.join(labeledFeaturesPath, file)
        fileContent = np.load(filePath, allow_pickle=True)
        
        currentFeature = fileContent.item()['features'].numpy()
        # TODO WTF IS THIS
        currentFeature = np.mean(currentFeature, axis=0)

        featuresLabeledSet[filePath] = currentFeature

    # Doing this twice is unfortunate, but it's an easy way to get all the confidence scores in a single array.
    # This way we can filter the number of samples in the unlabeled set to be only the ones where their confidence
    # score is < np.median(confidences)
    for file in os.listdir(unlabeledFeaturesPath):
        filePath    = os.path.join(unlabeledFeaturesPath, file)
        fileContent = np.load(filePath, allow_pickle=True)

        currentConfidence = fileContent.item()['confidence']

        confidences.append(currentConfidence)

    # Get the median confidence
    confidences      = np.asarray(confidences)
    medianConfidence = np.median(confidences)


    # Get the file paths for the features in the unlabeled set. Only add the samples where their confidence is < medianConfidence
    for file in os.listdir(unlabeledFeaturesPath):
        filePath    = os.path.join(unlabeledFeaturesPath, file)
        fileContent = np.load(filePath, allow_pickle=True)
        
        currentFeature    = fileContent.item()['features'].numpy()
        currentConfidence = fileContent.item()['confidence']

        if currentConfidence < medianConfidence:
            # TODO WTF IS THIS
            print(currentFeature.shape)
            currentFeature = np.mean(currentFeature, axis=0)
            print(currentFeature.shape)

            featuresUnlabeledSet[filePath] = (currentFeature, currentConfidence)


    return featuresLabeledSet, featuresUnlabeledSet



def rankSimiliratyByFeatures(featuresLabeledSet: dict, featuresUnlabeledSet: dict, strategy: str, saveFolder: str, nLabelings: int) -> dict:
    """
    Given a set of features for the labeled and unlabeled sets, this function selects the top 'nLabelings' best samples
    in the unlabeled set that are the most different from the samples in the labeled set.

    Args:
        featuresLabeledSet   (dict): The dict containing the features for the labeled set.   Key = video name, Value = features
        featuresUnlabeledSet (dict): The dict containing the features for the unlabeled set. Key = video name, Value = features, confidence
        strategy              (str): The selection strategy. Can be "cosine" or "kcenter".
        saveFolder            (str): The folder where to save the rank for best samples
        nLabelings            (int): How many samples should be selected from the unlabeled set.

    Returns:
        dict: The ranking for every sample in the unlabeled set.
    """
    # For now, each sample has a default similarity score of "0"
    similarityRank       = defaultdict(lambda: 0)

    if strategy == "cosine":
        # For every sample in the unlabeled set, compare it to every sample in the labeled set and save the highest similarity
        # value achieved by it in 'similarityRank'
        for nameUnlabeledFeature, (unlabeledFeature, unlabeledConfidence) in tqdm(featuresUnlabeledSet.items()):
            # Strip the string and the _features.npy suffix to get only the name of the video.
            # The processed string will be something like "01April_2010_Thursday_heute_default-3"
            nameUnlabeledFeature = nameUnlabeledFeature.split("/")[-1].removesuffix("_features.npy")
            
            for _, labeledFeature in featuresLabeledSet.items():
                # The similarity is the confidence score + the cosine similarity. This aggregates uncertainty and
                # representativeness into a single score!
                similarity = cosineSimilarity(unlabeledFeature, labeledFeature)

                if similarity > similarityRank[nameUnlabeledFeature]:
                    similarityRank[nameUnlabeledFeature] = similarity
        

        similarityRank = dict(sorted(similarityRank.items(), key=lambda x:x[1]))
5000


    elif strategy == "kcenter":
        newFeatUnlabeledSet = dict()
        for nameUnlabeledFeature, (unlabeledFeature, unlabeledConfidence) in tqdm(featuresUnlabeledSet.items()):
            # Strip the string and the _features.npy suffix to get only the name of the video.
            # The processed string will be something like "01April_2010_Thursday_heute_default-3"
            nameUnlabeledFeature = nameUnlabeledFeature.split("/")[-1].removesuffix("_features.npy")
            newFeatUnlabeledSet[nameUnl5000abeledFeature] = unlabeledFeature


        similarityRank = kCenter(newFeatUnlabeledSet, nLabelings)
        print(similarityRank)


    similarityRank = { i: np.nan for i in similarityRank.keys() }


    # Save the similarity ranking
    savePath = os.path.join(saveFolder, "SimilarityRank.json")
    with open(savePath, "w") as filePointer:
        json.dump(similarityRank, filePointer, indent=4)


    return similarityRank