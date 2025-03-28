from argparse import ArgumentParser
import os


def makeParser():
    """Creates a parser for the program parameters

    Returns:
        parser: an ArgumentParser object containing the program parameters
    """
    parser = ArgumentParser(description="Define the program parameters")

    parser.add_argument('-d','--dataset-path', type=str, required=True, 
                        help='The location of the dataset.')

    parser.add_argument('-l', '--n-labels', type=int, required=True,
                        help="The number of unlabeled samples that should be labeled each time the labeling step occurs. Total number of labeled samples will be --n-labels * --n-runs.")

    parser.add_argument('-r', '--n-runs', type=int, default=1,
                        help="How many loops the active learning framework will go through. This is important because it \
                        allows to save the results of individual sample selection loops, their respective training files, \
                        the generated glosses and their rankings all in individual folders. \
                        It's important to note that the total number of labeled samples will be --n-labels * --n-runs. \
                        Defaults to 1.")
    
    parser.add_argument('-w', '--work-dir', type=str, required=True,
                        help="Where to save the outputs for inference and training.")
    
    parser.add_argument('-c', '--custom-name', type=str, required=True,
                        help="A custom name for the experiment")
    
    parser.add_argument('-s', '--strategy', type=str, choices=["random", "active", "shortest", "longest"], required=True,
                        help="Whether to use active learning or random sampling.")

    parser.add_argument('--device', type=str, required=False, default=0,
                        help="What GPU to use during training and inference. Default=0.")

    return parser


def validateParams(args) -> None:
    """validates if the options passed on to the parser are valid

    Args:
        args (an argparse.Namespace object): contains the parameters in a easy to handle object
    """
    if args.n_labels <= 0:
        raise ValueError("Number of labels must be greater than 0.")

    if args.n_runs <= 0:
        raise ValueError("Number of runs must be greater than 0.")

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError("-d points to a path that does not exist.")
    
    
    # Convert args.dataset_path to absolute path and remove trailing "/"
    datasetPath          = args.dataset_path.rstrip("/")
    datasetPath          =  os.path.join(os.getcwd(), datasetPath)
    datasetParentFolder  =  "/".join(datasetPath.split("/")[ : -1])
    datasetName          =  datasetPath.split("/")[-1]
    args.dataset_path    =  datasetPath

    if os.path.exists(f"{datasetParentFolder}/{datasetName}-labeled"):
        raise FileExistsError(f"{datasetParentFolder}/{datasetName}-labeled already exists! Ideally this folder should be created by this program at a later point, meaning \
                              this is probably not your first time running this program. For safety reasons, this program will not continue and I \
                              ask you to deal with this before running the program again :)")

    if os.path.exists(f"{datasetParentFolder}/{datasetName}-unlabeled"):
        raise FileExistsError(f"{datasetParentFolder}/{datasetName}-unlabeled already exists! Ideally this folder should be created by this program at a later point, meaning \
                              this is probably not your first time running this program. For safety reasons, this program will not continue and I \
                              ask you to deal with this before running the program again :)")
    
    if os.path.exists(f"features"):
        raise FileExistsError(f"'features' folder already exists! This could lead to problems when extracting features, please make sure to delete it before running the code :)")