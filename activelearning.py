# Separar os dados de treino em 2 subconjuntos: treino unlabeled e treino labeled

# Passo 1:
    # Treinar o main.py com o conjunto de treino labeled

# Passo 2:
    # Salvar o melhor modelo

# Passo 3:
    # Rodar inferência do modelo no conjunto de treino unlabeled e add as 10% mais incertas ao conjunto de treino labeled

# Passo 4:
    # Voltar ao passo 1 e repetir até acabar as amostras


import subprocess
import logging
import random
import shutil
import yaml
import os

from transformers import XCLIPProcessor, XCLIPModel

from active_learning_modules.dataset_utils import splitDataset
from active_learning_modules.parser import makeParser, validateParams
from active_learning_modules.xclip_utils import parsePredictionsFile

logging.basicConfig(
    filename='./activelearning.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def preprocessRoutineWrapper(datasetPath: str, datasetName: str):
    """
    Runs the preprocess scrip to generatee the .stm groundtruth files, the gloss_dict.npy file, the [train, test, dev]_info.npy... Pretty much everything that we need to work with.
    
    It's important to note that, while the preprocessing routine in SlowFastSign generates the groundtruth for the train, test and dev files,
    they are not saved in the correct folder automatically. This makes it necessary to move the groundtruth files
    to their correct folder. This is what the second part of this function does

    Args:
        datasetPath (srt): The folder containing your data. It's the one with the phoenix-2014-multisigner/ folder.
        datasetName (str): The name of the dataset. Could be anything theoretically, but needs to match the one in the .yaml config file for the rest of the program to work
    """
    subprocess.run(f"cd preprocess; python3 dataset_preprocess.py --dataset {datasetName} --dataset-root {datasetPath}/phoenix-2014-multisigner", shell=True, check=True)

    for subset in ["train", "test", "dev"]:
        currentSubsetGroundtruthPath = os.path.join(f"./preprocess/{datasetName}", f"{datasetName}-groundtruth-{subset}.stm")
        shutil.copy(currentSubsetGroundtruthPath, "./evaluation/slr_eval/")
    


def labelDataPoints(labeledSubsetPath: str, unlabeledSubsetPath: str, nSamplesToLabel: int, selectedSamples=None, isFirstLabelingLoop=False):
    """Moves the selected data points from the unlabeled pool to the labeled pool. 

    Since we already have all the datapoints and this is a simulation, 
    we will simply move the selected data points from the unlabeledSubset to the labeledSubset.

    More specifically, the unlabeledSubset/phoenix-2014-multisigner/annotations/manual/test.corpus.csv file is our simulated unlabeled pool.
    Over there we have a bunch of data points that the training process hasn't seen yet. The active learning loop selected which of these are the ones that the model
    wants to learn from, and this function here is responsible for moving them to the labeled pool, so the next learning round incorporates them.

    Args:
        labeledSubsetPath         (str): The path to the labeled subset
        unlabeledSubsetPath       (str): The path to the unlabeled subset
        nSamplesToLabel           (int): How many data samples should be moved to the labeled pool.
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


    # If we're on the first labeling loop, this means that our labeled pool is empty and we have to create it!
    # To do this, first create the file and add the header to it.
    if isFirstLabelingLoop:
        with open(labeledPool, "w") as f:
            f.writelines(header) # Write the header because since this is the initial labeling, the labeledPool file is guaranteed to be empty!

        # The operation for the first labeling loop is the following: randomly select a couple of samples from the unlabeled pool. These will be moved to the labeled pool.
        random.shuffle(unlabeledPoolData)

        newlyLabeledInstances = unlabeledPoolData[0 : nSamplesToLabel]
        unlabeledInstances    = unlabeledPoolData[nSamplesToLabel : ]
    #TODO Finish this
    else:
        #for i in range(args.)
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
    
    labeledSubsetName   = labeledSubsetPath.split("/")[-1]
    unlabeledSubsetName = unlabeledSubsetPath.split("/")[-1]
    
    # Create config files for the unlabeled and labeled datasets
    for subsetPath, subsetName in zip([labeledSubsetPath, unlabeledSubsetPath], [labeledSubsetName, unlabeledSubsetName]):
        config = dict(  
                        dataset_root      = f"{subsetPath}/phoenix-2014-multisigner",
                        dict_path         = './preprocess/phoenix2014/gloss_dict.npy',
                        evaluation_dir    = './evaluation/slr_eval',
                        evaluation_prefix =f"{subsetName}-groundtruth"
                        )
        
        with open(f"./configs/{subsetName}.yaml", "w") as outfile:
            yaml.dump(config, outfile)

    # Now we do our first labeling loop. This will create a train.corpus.csv file for the labeled subset.
    labelDataPoints(labeledSubsetPath, unlabeledSubsetPath, args.n_labels, selectedSamples=None, isFirstLabelingLoop=True)
    
    # Run preprocess routine for labeled subset
    preprocessRoutineWrapper(labeledSubsetPath, labeledSubsetName)

    # Train on labeled subset
    subprocess.run(f"python3 main.py --device 0 --dataset {labeledSubsetName} --loss-weights Slow=0.25 Fast=0.25 --work-dir work_dir/{labeledSubsetName}", shell=True, check=True)
    
    # Now, we start the part of the Active Learning loop where we look for significant samples in the unlabeled subset
    
    # Run preprocess routine for the unlabeled subset
    preprocessRoutineWrapper(unlabeledSubsetPath, unlabeledSubsetName)

    # Run inference on the unlabeled set with the weights of the model that was just trained on the labeled set
    subprocess.run(f"python main.py --device 0 --dataset {unlabeledSubsetName} --phase test --load-weights ./work_dir/{labeledSubsetName}/_best_model.pt --work-dir ./work_dir/{unlabeledSubsetName} --enable-sample-selection", shell=True, check=True)


    predictionsFilePath = os.path.join(f"./work_dir/{unlabeledSubsetName}", "tmp2.stm")
    modelName           = "microsoft/xclip-base-patch32-16-frames"

    processor           = XCLIPProcessor.from_pretrained(modelName)
    model               = XCLIPModel.from_pretrained(modelName)

    # Inference to get similarity scores
    model.eval()

    videoPaths, glossPredictions = parsePredictionsFile(predictionsFilePath, unlabeledSubsetPath)

    # Now we find out which are the most informative predictions made for the unlabeled set. 
    mostInformativeSamples = findMostInformativeSamples(os.path.join("work_dir", unlabeledSubsetName), unlabeledSubsetPath)

    raise Exception
    labelDataPoints(labeledSubsetPath, unlabeledSubsetPath, args.n_labels, selectedSamples=mostInformativeSamples, isFirstLabelingLoop=False)