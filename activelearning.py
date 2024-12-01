# Separar os dados de treino em 2 subconjuntos: treino unlabeled e treino labeled

# Passo 1:
    # Treinar o main.py com o conjunto de treino labeled

# Passo 2:
    # Salvar o melhor modelo

# Passo 3:
    # Rodar inferência do modelo no conjunto de treino unlabeled e add as 10% mais incertas ao conjunto de treino labeled

# Passo 4:
    # Voltar ao passo 1 e repetir até acabar as amostras


import random
import shutil
import os

from argparse import ArgumentParser


def labelDataPoints(datasetPath, nLabelings, selectedSamples=None, isFirstLabelingLoop=False):
    """Moves the selected data points from the unlabeled pool to the labeled pool. 

    Since we already have all the datapoints and this is a simulation, 
    we will simply move the selected data points from the unlabeledSubset to the labeledSubset.

    More specifically, the unlabeledSubset/phoenix-2014-multisigner/annotations/manual/test.corpus.csv file is our simulated unlabeled pool.
    Over there we have a bunch of data points that the training process hasn't seen yet. The active learning loop selected which of these are the ones that the model
    wants to learn from, and this function here is responsible for moving them to the labeled pool, so the next learning round incorporates them.

    Args:
        datasetPath (str): The path to the dataset
        nLabelingsRandomSelection (int): How many data samples should be labeled randomly. Only used if selectedSamples=None.
        selectedSamples (list): The names of the samples that were selected to be labeled.
        isFirstLabelingLoop (bool): Whether or not it is the first labeling loop. This will create a new labeled pool file!
    """
    datasetPath     = os.path.join(os.getcwd(), datasetPath)
    labeledSubset   = f"{datasetPath}-labeled"
    unlabeledSubset = f"{datasetPath}-unlabeled"

    labeledPool   = os.path.join(labeledSubset, "phoenix-2014-multisigner/annotations/manual/train.corpus.csv")
    unlabeledPool = os.path.join(unlabeledSubset, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv")

    # Get everything in the unlabeled pool
    with open(unlabeledPool, "r") as f:
        # Isolate the header
        dataPoints  = f.readlines()
        header      = dataPoints[0]
        del dataPoints[0]

    
    # If we're on the first labeling loop, this means that our labeled pool is empty and we have to create it!
    # To do this, first create the file and add the header to it.
    if isFirstLabelingLoop:
        with open(labeledPool, "w") as f:
            f.writelines(header) # Write the header because since this is the initial labeling, the labeledPool file is guaranteed to be empty!
    

    # Now, if there are no selected samples, it can mean two things:
    #   * Either we are on the very first labeling loop and we have to create our labeled pool. This first labeled pool is created
    # by randomly selecting samples from the unlabeled pool and adding them to it.

    #   * Or we are testing the random sampling selection strategy.

    # Either way, the operation for both cases is the same: randomly select a couple of samples from the unlabeled pool. These will be moved to the labeled pool.
    if selectedSamples is None:
        random.shuffle(dataPoints)

        labeledInstances   = dataPoints[0 : nLabelings]
        unlabeledInstances = dataPoints[nLabelings : ]


    # On the other hand, if we know what samples have been selected to be labeled, we move them from the unlabeled pool to the labeled pool.
    else:
        # TODO FINISH THIS
        labeledInstances   = dataPoints[0 : nLabelings]
        unlabeledInstances = dataPoints[nLabelings : ]


    # When we 'label' an instance, we have to remove it from the unlabeled pool. It is easier to just delete the unlabeledPool file and write what we want
    # instead of figuring out what lines should be kept in or removed one by one
    with open(unlabeledPool, "w") as f:
        f.writelines(header)
        f.writelines(unlabeledInstances)

    # Next, we append the selected samples to the labeledPool file and we're done!
    with open(labeledPool, "a") as f:
        f.writelines(labeledInstances)



def make_parser():
    """Creates a parser for the program parameters

    Returns:
        parser: an ArgumentParser object containing the program parameters
    """
    parser = ArgumentParser(description="Define the program parameters")

    parser.add_argument('-d','--dataset-path', type=str, required=True, 
                        help='The location of the dataset.')

    parser.add_argument('-p', '--labeling-percentage', type=float, required=True,
                        help="The percentage of the dataset that will go to the labeled subset when splitting.")

    return parser



def validateParams(args):
    """validates if the options passed on to the parser are valid

    Args:
        args (an argparse.Namespace object): contains the parameters in a easy to handle object

    Raises:
        ValueError: _description_
        FileExistsError: _description_
        FileNotFoundError: _description_
        ValueError: _description_
        ValueError: _description_
        FileExistsError: _description_
        NotImplementedError: _description_
        FileNotFoundError: _description_
        ValueError: _description_
        FileNotFoundError: _description_
        ValueError: _description_
        FileExistsError: _description_
        FileExistsError: _description_
        ValueError: _description_
    """
    if args.labeling_percentage <= 0:
        raise ValueError("-p must be greater than 0.")

    elif args.labeling_percentage >= 1:
        raise ValueError("-p must be lesser than 1.")

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError("-d points to a path that does not exist.")
    
    pathForChecking = args.dataset_path
    for subfolder in ["phoenix-2014-multisigner", "annotations", "manual"]:
        pathForChecking = os.path.join(pathForChecking, subfolder)
        if not os.path.exists(pathForChecking):
            raise FileNotFoundError(f"{pathForChecking} does not exist! Make sure the structure inside the dataset folder matches the one in Phoenix2014!")

    if os.path.exists(f"{args.dataset_path}-labeled"):
        raise FileExistsError(f"{args.d}-labeled already exists! Ideally this folder should be created by this program at a later point, meaning \
                              this is probably not your first timerunning this program. For safety reasons, this program will not continue and I \
                              ask you to deal with this before running the program again :)")

    if os.path.exists(f"{args.dataset_path}-unlabeled"):
        raise FileExistsError(f"{args.dataset_path}-unlabeled already exists! Ideally this folder should be created by this program at a later point, meaning \
                              this is probably not your first timerunning this program. For safety reasons, this program will not continue and I \
                              ask you to deal with this before running the program again :)")


    print(f"Creating {args.dataset_path}-labeled and {args.dataset_path}-unlabeled now!")
    os.makedirs(f"{args.dataset_path}-labeled")
    os.makedirs(f"{args.dataset_path}-unlabeled")


def splitDataset(datasetPath):
    """Creates \"copies\" of datasetPath for serving as the labeled and the unlabeled subsets in the active learning loop.
    The structure copies the one used in the phoenix2014 dataset. To avoid copying all of the data, symlinks to the original dataset are used for
    each of the unlabeled and labeled subsets.

    Args:
        datasetPath (str): a path pointing to where the dataset is
    """
    
    datasetPath     = os.path.join(os.getcwd(), datasetPath)
    labeledSubset   = f"{datasetPath}-labeled"
    unlabeledSubset = f"{datasetPath}-unlabeled"

    for split in [labeledSubset, unlabeledSubset]:
        # Creates [labeledSubset, unlabeledSubset]/phoenix-2014-multisigner
        os.makedirs(os.path.join(split, "phoenix-2014-multisigner"))
        print(f"Created {os.path.join(split, 'phoenix-2014-multisigner')}")


        # Creates symlinks {datasetPath}/phoenix-2014-multisigner/[evaluation, features, models] -> [labeledSubset, unlabeledSubset]/phoenix-2014-multisigner/[evaluation, features, models]
        for subfolder in ["evaluation", "features", "models"]:
            originalMultisignerPath = os.path.join(datasetPath, f"phoenix-2014-multisigner/{subfolder}")
            newMultisignerPath      = os.path.join(split, f"phoenix-2014-multisigner/{subfolder}")
            os.symlink(originalMultisignerPath, newMultisignerPath)
            print(f"Created symlink {originalMultisignerPath} -> {newMultisignerPath}")


        # Creates folder [labeledSubset, unlabeledSubset]/phoenix-2014-multisigner/annotations/manual
        os.makedirs(os.path.join(split, "phoenix-2014-multisigner/annotations/manual"))
        print(f"Created {os.path.join(split, 'phoenix-2014-multisigner/annotations/manual')}", end="\n\n")


    trainAnnotation = os.path.join(datasetPath, "phoenix-2014-multisigner/annotations/manual/train.corpus.csv")
    testAnnotation  = os.path.join(datasetPath, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv")
    devAnnotation   = os.path.join(datasetPath, "phoenix-2014-multisigner/annotations/manual/dev.corpus.csv")

    # Original training annotation goes to the unlabeled subset as the 'test' file. It goes in as the test file because the test file is used for ]
    # running inference, and we want to run inference on the 'unlabeled' data points. Ideally it should be called something else because this gives the impression
    # that it is the original test annotation, and it might get confusing. Just keep in mind that this is our simulated unlabeled data pool, and it is only called 
    # test.corpus.csv because I don't want to change the code of SlowFastSign too much.
    shutil.copy(trainAnnotation, os.path.join(unlabeledSubset, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv"))

    # Original test and dev annotations go to the labeled set as themselves. We will be testing the performance on them, so they have to go in unmodified.
    shutil.copy(testAnnotation, os.path.join(labeledSubset, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv"))
    shutil.copy(devAnnotation,  os.path.join(labeledSubset, "phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"))


if __name__ == '__main__':
    args = make_parser().parse_args()
    #validateParams(args)

    #splitDataset(args.dataset_path)
    labelDataPoints(args.dataset_path, 10, None, False)