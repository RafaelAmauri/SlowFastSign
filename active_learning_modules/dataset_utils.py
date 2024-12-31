import logging
import shutil
import os


logging.basicConfig(
    filename='./activelearning.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def splitDataset(datasetPath: str) -> tuple[str, str]:
    """
    Creates copies of datasetPath for serving as the labeled and the unlabeled subsets in the active learning loop.
    The structure copies the one used in the phoenix2014 dataset. To avoid copying all of the data, symlinks to the original dataset are used for
    copying the images.

    Args:
        datasetPath (str): a path pointing to where the dataset is
    """
    
    datasetParentFolder  =  "/".join(datasetPath.split("/")[ : -1])
    datasetName          =  datasetPath.split("/")[-1]
    
    # Save the path for the labeled and unlabeled folders

    labeledSubsetPath   = os.path.join(datasetParentFolder, f"{datasetName}-labeled")
    unlabeledSubsetPath = os.path.join(datasetParentFolder, f"{datasetName}-unlabeled")

    copyDataset(datasetPath, labeledSubsetPath)
    copyDataset(datasetPath, unlabeledSubsetPath)

    # Save where the original annotation files are
    trainAnnotationPath = os.path.join(datasetPath, "phoenix-2014-multisigner/annotations/manual/train.corpus.csv")
    testAnnotationPath  = os.path.join(datasetPath, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv")
    devAnnotationPath   = os.path.join(datasetPath, "phoenix-2014-multisigner/annotations/manual/dev.corpus.csv")


    # The original train.corpus.csv annotation goes to the unlabeled subset as the test.corpus.csv file. It goes in as the test file because the test file is used for
    # running inference, and we want to run inference on the 'unlabeled' data points. Ideally it should be called something else because this gives the impression
    # that it is the original test annotation, and it might get confusing. Just keep in mind that this is our simulated unlabeled data pool, and it is only called 
    # test.corpus.csv because I don't want to change the code of SlowFastSign too much.
    shutil.copy(trainAnnotationPath, os.path.join(unlabeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv"))

    # Since we now have a train.corpus.csv file that was 'swapped' for the test.corpus.csv file inside the unlabeled folder, we now have to also swap the
    # corresponding train and test folders inside the unlabeledSubsetPath/phoenix-2014-multisigner/features folder to reflect this change!

    # Now,    unlabeledSubsetPath/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/train
    # becomes unlabeledSubsetPath/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/test

    # And unlabeledSubsetPath/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/test
    # becomes unlabeledSubsetPath/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/train
    unlabeledSubsetFeaturesFolder = os.path.join(unlabeledSubsetPath, "phoenix-2014-multisigner/features")
    for fullFrameFolderCategory in ["fullFrame-210x260px", "fullFrame-256x256px"]:
        trainFullFrameCategoryFolder = os.path.join(unlabeledSubsetFeaturesFolder, f"{fullFrameFolderCategory}/train")
        testFullFrameCategoryFolder  = os.path.join(unlabeledSubsetFeaturesFolder, f"{fullFrameFolderCategory}/test")
        
        shutil.move(trainFullFrameCategoryFolder, "./tmp")
        shutil.move(testFullFrameCategoryFolder, trainFullFrameCategoryFolder)
        shutil.move("./tmp", testFullFrameCategoryFolder)

    # The SlowFastSign script requires a test, dev and train annotation file in every dataset, even if they are empty. Right now we only have a test file in the unlabeled set, so now we 
    # create empty dev and train files.
    header = "id|folder|signer|annotation"
    with open(os.path.join(unlabeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"), "w") as outfile:
        outfile.writelines(header)
    
    with open(os.path.join(unlabeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/train.corpus.csv"), "w") as outfile:
        outfile.writelines(header)


    # Original test and dev annotations go to the labeled set as themselves. We will be testing the performance on them, so they have to go in unmodified.
    shutil.copy(testAnnotationPath, os.path.join(labeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv"))
    shutil.copy(devAnnotationPath,  os.path.join(labeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"))


    return labeledSubsetPath, unlabeledSubsetPath


def copyDataset(datasetPath, newDatasetPath, copyAnnotations=False) -> None:
    """
    Originally meant as a part of the splitDataset function, but became more modular to allow creating copies of everything. 
    Now, you just pass a name and it will create a copy of the original with that name.

    The structure copies the one used in the phoenix2014 dataset. To avoid copying all of the data, symlinks to the original dataset are used for
    copying the images. 
    
    The main difference from just using cp -r is that cp -r would make a copy of everything, and this function smartly uses symlinks in larger folders
    with images to avoid making unecessary copies.

    Args:
        datasetPath     (str): a path pointing to where the dataset is
        newDatasetPath  (str): Where the copy is going to be created
        copyAnnotations (bool): Whether to copy the original annotation files to the new directory as well. Usually not recommended because
                                it might mess things up when splitDataset() tries to set up the labeled and unlabeled pools. But if what
                                you want is simply to make a perfect copy of datasetPath, then you can enable it without problems!
    """

    # These will all be created by the script automatically. After they're created, we will create symlinks to the relevant files and folders.
    internalFolders = [
                        "phoenix-2014-multisigner/annotations/manual",
                        "phoenix-2014-multisigner/evaluation",
                        "phoenix-2014-multisigner/features/fullFrame-210x260px",
                        "phoenix-2014-multisigner/features/fullFrame-256x256px"
                        ]


    for folder in internalFolders:
        os.makedirs(os.path.join(newDatasetPath, folder))

    
    # Creates symlinks {datasetPath}/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/[train, test, dev] -> 
    # -> {newDatasetPath}/phoenix-2014-multisigner/features/[fullFrame-210x260px, fullFrame-256x256px]/[train, test, dev]
    for fullFrameFolderCategory in ["fullFrame-210x260px", "fullFrame-256x256px"]:
        originalFullFrameFolder = os.path.join(datasetPath, f"phoenix-2014-multisigner/features/{fullFrameFolderCategory}")
        newFullFrameFolder      = os.path.join(newDatasetPath, f"phoenix-2014-multisigner/features/{fullFrameFolderCategory}")

        for subset in ["train", "test", "dev"]:
            originalSubsetPath = os.path.join(originalFullFrameFolder, subset)
            newSubsetPath      = os.path.join(newFullFrameFolder, subset)
                
            os.symlink(originalSubsetPath, newSubsetPath)


    # Copies {datasetPath}/phoenix-2014-multisigner/annotations/manual/[dev,test].corpus.csv -> {newDatasetPath}/phoenix-2014-multisigner/annotations/manual/[dev,test].corpus.csv
    if copyAnnotations:
        for split in ["dev", "test"]:
            originalAnnotationPath   = os.path.join(datasetPath, f"phoenix-2014-multisigner/annotations/manual/{split}.corpus.csv")
            newDatasetAnnotationPath = os.path.join(newDatasetPath, f"phoenix-2014-multisigner/annotations/manual/{split}.corpus.csv")
            
            shutil.copy(originalAnnotationPath, newDatasetAnnotationPath)