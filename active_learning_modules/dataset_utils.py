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


    # The original training annotation goes to the unlabeled subset as the 'test' file. It goes in as the test file because the test file is used for
    # running inference, and we want to run inference on the 'unlabeled' data points. Ideally it should be called something else because this gives the impression
    # that it is the original test annotation, and it might get confusing. Just keep in mind that this is our simulated unlabeled data pool, and it is only called 
    # test.corpus.csv because I don't want to change the code of SlowFastSign too much.
    shutil.copy(trainAnnotationPath, os.path.join(unlabeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv"))

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

    # Creates {newDatasetPath}/phoenix-2014-multisigner
    os.makedirs(os.path.join(newDatasetPath, "phoenix-2014-multisigner"))
    logging.info(f"Created {os.path.join(newDatasetPath, 'phoenix-2014-multisigner')}")


    # Creates symlinks {datasetPath}/phoenix-2014-multisigner/[evaluation, features, models] -> {newDatasetPath}/phoenix-2014-multisigner/[evaluation, features, models]
    for subfolder in ["evaluation", "features", "models"]:
        originalMultisignerPath = os.path.join(datasetPath, f"phoenix-2014-multisigner/{subfolder}")
        newMultisignerPath      = os.path.join(newDatasetPath, f"phoenix-2014-multisigner/{subfolder}")
        os.symlink(originalMultisignerPath, newMultisignerPath)
        logging.info(f"Created symlink {originalMultisignerPath} -> {newMultisignerPath}")


    # Creates folder {newDatasetPath}/phoenix-2014-multisigner/annotations/manual
    os.makedirs(os.path.join(newDatasetPath, "phoenix-2014-multisigner/annotations/manual"))
    logging.info(f"Created {os.path.join(newDatasetPath, 'phoenix-2014-multisigner/annotations/manual')}")

    # Copies {datasetPath}/phoenix-2014-multisigner/annotations/manual/[dev,test].corpus.csv -> {newDatasetPath}/phoenix-2014-multisigner/annotations/manual/[dev,test].corpus.csv
    if copyAnnotations:
        for split in ["dev", "test"]:
            originalAnnotationPath   = os.path.join(datasetPath, f"phoenix-2014-multisigner/annotations/manual/{split}.corpus.csv")
            newDatasetAnnotationPath = os.path.join(newDatasetPath, f"phoenix-2014-multisigner/annotations/manual/{split}.corpus.csv")
            
            shutil.copy(originalAnnotationPath, newDatasetAnnotationPath)