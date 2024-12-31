import os
import json
import torch

from tqdm import tqdm
from queue import PriorityQueue
from collections import OrderedDict


from active_learning_modules.xclip_utils import getKeyFramesFromFolder


def xclipInference(model, videoPaths, glossPredictions, predictionsWorkDir):
    keyFrames         = []
    predictionRanking = PriorityQueue()
    for vPath, predictedGloss in zip(videoPaths, glossPredictions):
        # X-Clip can only handle 8 frames or 16 frames at a time, so we need to choose the 16 most significant frames :(
        keyFrames = getKeyFramesFromFolder(vPath, 16)

        # We give the model the chance to choose between the gloss that the other model generated or some random garbage.
        # If it chooses the random garbage with higher probability, it means that this could be a good candidate for labeling.
        possibleOptions = [predictedGloss, "_"]

        print(possibleOptions, predictedGloss)
        inputs = processor(text=possibleOptions, videos=keyFrames, return_tensors="pt", padding=True) # type: ignore
    
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted class probabilities
        probs = outputs.logits_per_video.softmax(dim=1)

        predictedLabel = possibleOptions[torch.argmax(probs)]
        predictionProbability = torch.max(probs)

        predictionRanking.put((vPath, predictionProbability))
    
    
    # Write to disk
    content = OrderedDict()
    while not predictionRanking.empty():
        path, confidence = predictionRanking.get()
        confidence = confidence.item()
        content[path] = confidence
    
    with open(os.path.join(predictionsWorkDir, "predictionRankings.json"), "w") as f:
        json.dump(content, f)

    return content


def trainXclip():
    pass