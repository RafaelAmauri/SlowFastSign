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
import torch
import torch.utils
import torch.utils.data
import yaml
import os
import gc

from transformers import XCLIPProcessor, XCLIPModel

from active_learning_modules.dataset_utils import splitDataset, copyDataset
from active_learning_modules.parser import makeParser, validateParams
from active_learning_modules.xclip import trainXClip, alignmentVideoGeneratedGloss
from active_learning_modules.xclip_utils import parseSlowFastSignPredictionsFile, parseAnnotationFile, parseInformativenessRanking
from active_learning_modules.videoglossdataset import VideoGlossDataset, videoGlossDatasetCollateFn


logging.basicConfig(
    filename='./activelearning.log',
    level=logging.INFO,
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
    


def labelDataPoints(labeledSubsetPath: str, unlabeledSubsetPath: str, nSamplesToLabel: int, selectedSamples: list, isFirstLabelingLoop=False):
    """Moves the selected data points from the unlabeled pool to the labeled pool. 

    Since we already have all the datapoints and this is a simulation, 
    we will simply move the selected data points from the unlabeledSubset to the labeledSubset.

    More specifically, the unlabeledSubset/phoenix-2014-multisigner/annotations/manual/test.corpus.csv file is our simulated unlabeled pool.
    Over there we have a bunch of data points that the training process hasn't seen yet. The active learning loop selected which of these are the ones that the model
    wants to learn from, and this function here is responsible for moving them to the labeled pool, so the next learning round incorporates them.

    Args:
        labeledSubsetPath          (str): The path to the labeled subset
        unlabeledSubsetPath        (str): The path to the unlabeled subset
        nSamplesToLabel            (int): How many data samples should be moved to the labeled pool.
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

        newlyLabeledData        = unlabeledPoolData[0 : nSamplesToLabel]
        remainingUnlabeledData  = unlabeledPoolData[nSamplesToLabel : ]
    
    # If it is not the first labeling loop, simply pick the selected samples from the unlabeled pool.
    else:
        newlyLabeledData        = []
        remainingUnlabeledData  = unlabeledPoolData.copy()
        
        for unlabeledSample in unlabeledPoolData:
            unlabeledFolderName = unlabeledSample.split("|")[0]

            if unlabeledFolderName in selectedSamples:
                print(unlabeledFolderName)
                newlyLabeledData.append(unlabeledSample)
                remainingUnlabeledData.remove(unlabeledSample)

    
    # When we 'label' an instance, we have to remove it from the unlabeled pool. It is easier to just delete the unlabeledPool file and write what we want
    # instead of figuring out what lines should be kept in or removed one by one
    with open(unlabeledPool, "w") as f:
        f.writelines(header)
        f.writelines(remainingUnlabeledData)

    # Next, we append the selected samples to the labeledPool file and we're done!
    with open(labeledPool, "a") as f:
        f.writelines(newlyLabeledData)
    


if __name__ == '__main__':
    args = makeParser().parse_args()
    validateParams(args)

    runId = 1
    # Creates a labeled and unlabeled subset of the dataset. They are called {datasetParentFolder}/{datasetName}-[labeled, unlabeled]-run{runId}
    labeledSubsetPath, unlabeledSubsetPath = splitDataset(args.dataset_path, args.custom_name, runId)
    
    labeledSubsetName   = labeledSubsetPath.split("/")[-1]
    unlabeledSubsetName = unlabeledSubsetPath.split("/")[-1]

    # Now we do our first labeling loop. This will create a train.corpus.csv file for the labeled subset.
    labelDataPoints(labeledSubsetPath, unlabeledSubsetPath, args.n_labels, selectedSamples=[], isFirstLabelingLoop=True)
    
    for runId in range(1, args.n_runs+1):
        print(f"Starting Run {runId} now....")
        
        
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
        
        # Run preprocess routine for labeled subset
        preprocessRoutineWrapper(labeledSubsetPath, labeledSubsetName)

        # Train gloss generator on labeled subset
        subprocess.run(f"python3 main.py --device {args.device} --dataset {labeledSubsetName} --loss-weights Slow=0.25 Fast=0.25 --work-dir {args.work_dir}/{labeledSubsetName}", shell=True, check=True)

        # Prepare to train X-Clip on the labeled set
        if args.x_clip_n_frames == 16:
            modelName  = f"microsoft/xclip-base-patch32-16-frames"
        else:
            modelName  = f"microsoft/xclip-base-patch32"
        processor  = XCLIPProcessor.from_pretrained(modelName)
        model      = XCLIPModel.from_pretrained(modelName)

        labeledTrainGroundTruthByVideoPath = parseAnnotationFile(labeledSubsetPath, "train")
        labeledTrainVideoGlossDataset      = VideoGlossDataset(labeledTrainGroundTruthByVideoPath, nFrames=args.x_clip_n_frames)
        labeledTrainDataloader             = torch.utils.data.DataLoader(labeledTrainVideoGlossDataset, 
                                                                        batch_size=args.x_clip_batch_size, 
                                                                        shuffle=True, 
                                                                        collate_fn=lambda batch: videoGlossDatasetCollateFn(batch, processor)
                                                                        )

        trainXClip(model=model, 
                   processor=processor,
                   nEpochs=args.x_clip_epochs,
                   dataloader=labeledTrainDataloader,
                   saveFolder=f"{args.work_dir}/{labeledSubsetName}",
                   device=f"cuda:{args.device}"
                   )

        # Remove X-Clip model from VRAM 
        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()

        # Now, we start the part of the Active Learning loop where we look for significant samples in the unlabeled subset    
        # Run preprocess routine for the unlabeled subset
        preprocessRoutineWrapper(unlabeledSubsetPath, unlabeledSubsetName)
        
        # Run inference on the unlabeled set with the weights of the model that was just trained on the labeled set
        subprocess.run(f"python main.py --device {args.device} --dataset {unlabeledSubsetName} --phase test --load-weights {args.work_dir}/{labeledSubsetName}/_best_model.pt --work-dir {args.work_dir}/{unlabeledSubsetName} --enable-sample-selection", shell=True, check=True)
        
        predictionsFilePath = os.path.join(f"{args.work_dir}/{unlabeledSubsetName}", "tmp2.ctm")
        glossesByVideoPath  = parseSlowFastSignPredictionsFile(predictionsFilePath, unlabeledSubsetPath)
        
        processor  = XCLIPProcessor.from_pretrained(f"{args.work_dir}/{labeledSubsetName}/fine_tuned_xclip_processor")
        model      = XCLIPModel.from_pretrained(f"{args.work_dir}/{labeledSubsetName}/fine_tuned_xclip_model")

        # Now we find out which are the most informative predictions made for the unlabeled set.
        mostInformativeSamples = alignmentVideoGeneratedGloss(model, processor, glossesByVideoPath, args.x_clip_n_frames, f"{args.work_dir}/{unlabeledSubsetName}", "cuda")

        # Get only the args.n_labels most informative ones and format the dictionary into a list so its easier to match entries with the unlabeled pool.
        mostInformativeSamples = [key.split("/")[-2] for key in list(mostInformativeSamples.keys())[ : args.n_labels]]
        
        newLabeledSubsetPath   = labeledSubsetPath.rstrip(str(runId))   + str(runId+1)
        newUnlabeledSubsetPath = unlabeledSubsetPath.rstrip(str(runId)) + str(runId+1)

        copyDataset(labeledSubsetPath,   newLabeledSubsetPath,   copyAnnotations=True)
        copyDataset(unlabeledSubsetPath, newUnlabeledSubsetPath, copyAnnotations=True)

        labeledSubsetPath   = newLabeledSubsetPath
        labeledSubsetName   = labeledSubsetPath.split("/")[-1]
        unlabeledSubsetPath = newUnlabeledSubsetPath
        unlabeledSubsetName = unlabeledSubsetPath.split("/")[-1]

        print(mostInformativeSamples)
        labelDataPoints(labeledSubsetPath, unlabeledSubsetPath, args.n_labels, selectedSamples=mostInformativeSamples, isFirstLabelingLoop=False)

        del processor
        del model
        gc.collect()
        torch.cuda.empty_cache()