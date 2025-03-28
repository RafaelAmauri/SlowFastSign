import os
import json

from collections import defaultdict


def selectLongestVideos(unlabeledSubsetPath: str, saveFolder: str) -> dict:
    """Reads all the videos in the unlabeled set and selects the ones with the most amount of frames

    Args:
        unlabeledSubsetPath (str): The path to the unlabeled set

    Returns:
        dict: The 
    """
    videoLength = defaultdict(lambda: 0)

    with open(os.path.join(unlabeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv"), "r") as f:
        lines = f.readlines()
        del lines[0]

    for l in lines:
        videoID = l.split("|")[0]

        framesFolder = os.path.join(unlabeledSubsetPath, f"phoenix-2014-multisigner/features/fullFrame-256x256px/test/{videoID}/1")
        videoLength[videoID] = len(os.listdir(framesFolder))
    
    videoLength = dict(sorted(videoLength.items(), key=lambda x:x[1], reverse=True))
    
    # Save the similarity ranking
    savePath = os.path.join(saveFolder, "SimilarityRank.json")
    with open(savePath, "w") as filePointer:
        json.dump(videoLength, filePointer, indent=4)
    

    return videoLength



def selectShortestVideos(unlabeledSubsetPath: str, saveFolder: str) -> dict:
    """Reads all the videos in the unlabeled set and selects the ones with the least amount of frames

    Args:
        unlabeledSubsetPath (str): The path to the unlabeled set

    Returns:
        dict: The 
    """
    videoLength = defaultdict(lambda: 0)

    with open(os.path.join(unlabeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv"), "r") as f:
        lines = f.readlines()
        del lines[0]

    for l in lines:
        videoID = l.split("|")[0]

        framesFolder = os.path.join(unlabeledSubsetPath, f"phoenix-2014-multisigner/features/fullFrame-256x256px/test/{videoID}/1")
        videoLength[videoID] = len(os.listdir(framesFolder))
    
    videoLength = dict(sorted(videoLength.items(), key=lambda x:x[1]))

    savePath = os.path.join(saveFolder, "SimilarityRank.json")
    with open(savePath, "w") as filePointer:
        json.dump(videoLength, filePointer, indent=4)
    

    return videoLength
