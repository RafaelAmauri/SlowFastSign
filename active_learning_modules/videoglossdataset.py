import torch

from active_learning_modules.xclip_utils import getKeyFramesFromFolder

class VideoGlossDataset(torch.utils.data.Dataset): # type: ignore
    def __init__(self, groundTruthByVideoPath: dict, nFrames: int):
        """
        Args:
            groundTruthByVideoPath (dict): A dict that uses the video paths as keys to index the ground-truth of the dataset.
            processor     (CLIPProcessor): Processor to handle both text and image data.
        """
        self.videoPaths = list(groundTruthByVideoPath.keys())
        self.glosses    = list(groundTruthByVideoPath.values())
        self.nFrames    = nFrames

    def __len__(self):
        return len(self.videoPaths)


    def __getitem__(self, idx):
        # Load video
        video_path = self.videoPaths[idx]
        gloss      = self.glosses[idx]
        
        # Get keyframes from video
        keyFrames  = getKeyFramesFromFolder(video_path, self.nFrames)
        
        return keyFrames, gloss


def videoGlossDatasetCollateFn(batch, processor):
    # Initialize empty lists for the batch tensors
    allVideos = []
    allGlosses = []
    
    # Loop through the videos and glosses and add them to a their lists
    for video, gloss in batch:
        allVideos.append(video)
        allGlosses.append(gloss)

    # Use CLIPProcessor to preprocess the batch of texts and frames
    inputs = processor(text=allGlosses, videos=allVideos, return_tensors="pt", padding=True, truncation=True)

    return inputs