from argparse import ArgumentParser
import logging
import os


logging.basicConfig(
    filename='./activelearning.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def makeParser():
    """Creates a parser for the program parameters

    Returns:
        parser: an ArgumentParser object containing the program parameters
    """
    parser = ArgumentParser(description="Define the program parameters")

    parser.add_argument('-d','--dataset-path', type=str, required=True, 
                        help='The location of the dataset.')

    parser.add_argument('-l', '--n-labels', type=int, required=True,
                        help="The number of unlabeled samples that should be labeled each time the labeling step occurs.")

    parser.add_argument('-m', '--mode', type=str, required=True, choices=["random", "active"],
                        help="The if active, uses the new architecture for active learning. If random, uses random sampling.")

    return parser


def validateParams(args) -> None:
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
    if args.n_labels <= 0:
        raise ValueError("Number of labels must be greater than 0.")

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