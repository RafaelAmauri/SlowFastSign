import os
import numpy as np

from PIL import Image


def parseAnnotationFile(subsetPath: str, mode: str):

    annotationFilePath = os.path.join(subsetPath, f"phoenix-2014-multisigner/annotations/manual/{mode}.corpus.csv")

    with open(annotationFilePath, "r") as f:
        lines  = f.readlines()
        del lines[0]

    videoPaths  = []
    groundTruth = []
    for line in lines:
        line = line.split("|")

        currentVideo         = line[0]
        currentVideoFullPath = os.path.join(f"{subsetPath}/phoenix-2014-multisigner/features/fullFrame-256x256px/{mode}", f"{currentVideo}/1")
        
        currentGloss = line[-1]
        currentGloss = currentGloss.rstrip("\n")
        
        videoPaths.append(currentVideoFullPath)
        groundTruth.append(currentGloss)


    return videoPaths, groundTruth


def getKeyFramesFromFolder(path, nFrames):
    files = os.listdir(path)
    files.sort()

    frames = []
    for file in files:
        img = np.array(Image.open(os.path.join(path, file)))
        
        frames.append(img)

    indices   = np.linspace(0, len(frames)-1, num=nFrames, dtype=np.int32)
    keyFrames = [ frames[idx] for idx in indices]

    return keyFrames


def parsePredictionsFile(predictionsFilePath: str, unlabeledSubsetPath: str):
    # Predictions are stored like this in the file:
    
    # 01April_2010_Thursday_heute_default-0 1 0.00 0.01 DAS-IST-ES
    # 01April_2010_Thursday_heute_default-0 1 0.01 0.02 ABEND
    # 01April_2010_Thursday_heute_default-0 1 0.02 0.03 __OFF__
    # 01April_2010_Thursday_heute_default-0 1 0.03 0.04 VERAENDERN
    # 01April_2010_Thursday_heute_default-0 1 0.04 0.05 __OFF__
    # 01April_2010_Thursday_heute_default-0 1 0.05 0.06 ABEND
    # 01April_2010_Thursday_heute_default-0 1 0.06 0.07 __OFF__
    # 01April_2010_Thursday_heute_default-0 1 0.07 0.08 ABEND
    # 01April_2010_Thursday_heute_default-0 1 0.08 0.09 __OFF__
    
    # 01August_2011_Monday_heute_default-7 1 0.00 0.01 __OFF__
    # 01August_2011_Monday_heute_default-7 1 0.01 0.02 ABEND
    # 01August_2011_Monday_heute_default-7 1 0.02 0.03 __OFF__
    # 01August_2011_Monday_heute_default-7 1 0.03 0.04 ABEND
    # 01August_2011_Monday_heute_default-7 1 0.04 0.05 __OFF__
    # 01August_2011_Monday_heute_default-7 1 0.05 0.06 ABEND
    # 01August_2011_Monday_heute_default-7 1 0.06 0.07 __OFF__
    # 01August_2011_Monday_heute_default-7 1 0.07 0.08 ABEND
    # 01August_2011_Monday_heute_default-7 1 0.08 0.09 __OFF__
    # 01August_2011_Monday_heute_default-7 1 0.09 0.10 ABEND
    # 01August_2011_Monday_heute_default-7 1 0.10 0.11 __OFF__

    # For each folder containing a video, there are several lines, each with an individual word.
    # We have to concatenate these words into a single list, so they can be easily passed over to the X-Clip text processor.
    with open(predictionsFilePath, "r") as f:
        lines            = f.readlines()
        videoPaths       = []
        videoPathsCheck  = set()
        glossPredictions = []

        previousUnlabeledSampleFolderName = -1

        for line in lines:
            line = line.split(" ")

            unlabeledSampleFolderName = line[0]
            predictedWord             = line[-1].rstrip("\n")

            # I am so sorry for doing this.
            unlabeledSampleFolderPath = os.path.join(f"{unlabeledSubsetPath}/phoenix-2014-multisigner/features/fullFrame-256x256px/test", f"{unlabeledSampleFolderName}/1")
            
            if unlabeledSampleFolderPath not in videoPathsCheck:
                videoPaths.append(unlabeledSampleFolderPath)
                videoPathsCheck.add(unlabeledSampleFolderPath)

            # If the we are still processing the predictions for the same folder
            if unlabeledSampleFolderName == previousUnlabeledSampleFolderName:
                glossPredictions[-1] += f"{predictedWord} "
            # if we are not
            else:
                if len(glossPredictions) != 0:
                    glossPredictions[-1] = glossPredictions[-1].rstrip(" ") # Remove trailing ' ' character.
                glossPredictions.append(f"{predictedWord} ")

            previousUnlabeledSampleFolderName = unlabeledSampleFolderName

    return videoPaths, glossPredictions

