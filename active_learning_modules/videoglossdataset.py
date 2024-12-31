import torch

from active_learning_modules.xclip_utils import getKeyFramesFromFolder

class VideoGlossDataset(torch.utils.data.Dataset):
    def __init__(self, videoPaths, glosses, nFrames):
        """
        Args:
            videoPaths (list): List of paths to video files.
            glosses (list): List of glosses corresponding to each video.
            processor (CLIPProcessor): Processor to handle both text and image data.
        """
        self.videoPaths = videoPaths
        self.glosses    = glosses
        self.nFrames    = nFrames

    def __len__(self):
        return len(self.videoPaths)


    def __getitem__(self, idx):
        # Load video
        video_path = self.videoPaths[idx]
        gloss      = self.glosses[idx]
        
        # Get keyframes from video
        keyFrames  = getKeyFramesFromFolder(video_path, self.nFramesnFrames)
        
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