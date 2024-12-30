# Separar os dados de treino em 2 subconjuntos: treino unlabeled e treino labeled

# Passo 1:
    # Treinar o main.py com o conjunto de treino labeled

# Passo 2:
    # Salvar o melhor modelo

# Passo 3:
    # Rodar inferência do modelo no conjunto de treino unlabeled e add as 10% mais incertas ao conjunto de treino labeled

# Passo 4:
    # Voltar ao passo 1 e repetir até acabar as amostras


import logging
import random
import os

from active_learning_modules.parser import makeParser, validateParams
from active_learning_modules.dataset_utils import copyDataset, splitDataset


logging.basicConfig(
    filename='./activelearning.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def labelDataPoints(labeledSubsetPath: str, unlabeledSubsetPath: str, nLabelings: int, selectedSamples=None, isFirstLabelingLoop=False):
    """Moves the selected data points from the unlabeled pool to the labeled pool. 

    Since we already have all the datapoints and this is a simulation, 
    we will simply move the selected data points from the unlabeledSubset to the labeledSubset.

    More specifically, the unlabeledSubset/phoenix-2014-multisigner/annotations/manual/test.corpus.csv file is our simulated unlabeled pool.
    Over there we have a bunch of data points that the training process hasn't seen yet. The active learning loop selected which of these are the ones that the model
    wants to learn from, and this function here is responsible for moving them to the labeled pool, so the next learning round incorporates them.

    Args:
        labeledSubsetPath         (str): The path to the labeled subset
        unlabeledSubsetPath       (str): The path to the unlabeled subset
        nLabelingsRandomSelection (int): How many data samples should be labeled randomly. Only used if selectedSamples=None.
        selectedSamples           (list): The names of the samples that were selected to be labeled.
        isFirstLabelingLoop       (bool): Whether or not it is the first labeling loop. This will create a new labeled pool file!
    """

    labeledPool   = os.path.join(labeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/train.corpus.csv")
    unlabeledPool = os.path.join(unlabeledSubsetPath, "phoenix-2014-multisigner/annotations/manual/test.corpus.csv")

    # Get everything in the unlabeled pool
    with open(unlabeledPool, "r") as f:
        # Isolate the header
        unlabeledPoolData  = f.readlines()
        header             = unlabeledPoolData[0]
        del unlabeledPoolData[0]

    #TODO Finish this!
    # If we're on the first labeling loop, this means that our labeled pool is empty and we have to create it!
    # To do this, first create the file and add the header to it.
    if isFirstLabelingLoop:
        with open(labeledPool, "w") as f:
            f.writelines(header) # Write the header because since this is the initial labeling, the labeledPool file is guaranteed to be empty!

            # The operation for the first labeling loop is the following: randomly select a couple of samples from the unlabeled pool. These will be moved to the labeled pool.
            random.shuffle(unlabeledPoolData)

            newlyLabeledInstances = unlabeledPoolData[0 : nLabelings]
            unlabeledInstances    = unlabeledPoolData[nLabelings : ]
    else:
        newlyLabeledInstances = selectedSamples
        unlabeledInstances    = unlabeledPoolData - selectedSamples

    # When we 'label' an instance, we have to remove it from the unlabeled pool. It is easier to just delete the unlabeledPool file and write what we want
    # instead of figuring out what lines should be kept in or removed one by one
    with open(unlabeledPool, "w") as f:
        f.writelines(header)
        f.writelines(unlabeledInstances)

    # Next, we append the selected samples to the labeledPool file and we're done!
    with open(labeledPool, "a") as f:
        f.writelines(newlyLabeledInstances)


if __name__ == '__main__':
    args = makeParser().parse_args()
    validateParams(args)

    
    datasetParentFolder  =  "/".join(args.dataset_path.split("/")[ : -1])
    datasetName          =  args.dataset_path.split("/")[-1]

    # Creates a labeled and unlabeled subset of the dataset. They are called {datasetParentFolder}/{datasetName}-[labeled, unlabeled]
    labeledSubsetPath, unlabeledSubsetPath = splitDataset(args.dataset_path)

    # Now we do our first labeling loop. This will create a train.corpus.csv file for the labeled subset.
    labelDataPoints(labeledSubsetPath, unlabeledSubsetPath, 10, selectedSamples=None, isFirstLabelingLoop=True)
    
    #trainModel(args.dataset_path)