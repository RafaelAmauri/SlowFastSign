import numpy as np

from tqdm import tqdm
from collections import defaultdict


def distance(vector1, vector2):
    """
    Calculated the eucledian distance between two N-dimensional vectors.

    Args:
        vector1 (numpy.typing.ArrayLike): An N-dimensional vector
        vector2 (numpy.typing.ArrayLike): An N-dimensional vector

    Returns:
        np.typing.ArrayLike: The eucledian distance between every 
    """
    return np.sqrt(np.sum((vector1 - vector2) ** 2, axis=1))


# TODO
# Demorou 5 horas para rodar e não terminou, tive que dar ctrl + C pra parar. Por favor, otimizar essa bagaça.
# Opções: * Clusterizar os embeddings
#         * Paralelizar o código de alguma maneira (DONE! - 6x speedup)
def kCenter(videoFeatures) -> dict:
    """
    Finds K different centers in an N-dimensional space.

    This is an implementation of the Coreset Active Learning algorithm.
    
    
    Coreset works as a farthest-first selection. 
    
    The algorithm:
      1. Picks an arbitrary starting point.
      2. For each remaining point, calculates its distance to the nearest selected center.
      3. Selects the point with the maximum minimum distance.
      4. Repeats until k centers are chosen.
    

    Args:
        videoFeatures (dict): A dictionary where the key is the video Id, and the value are the features for that video.

    Returns:
        dict: A dict where the key is the video Id, and the value is the ranking of said video.
    """
    # Each frame embedding is a vector of 2308 dimensions, but just in case I'll leave this here to keep
    # the code adaptable.
    frameFeatureDimensions = list(videoFeatures.values())[0].shape[-1]

    # The origin is just an array of zeros with the same shape as each frame. If each frame
    # is indeed 2308 dimensions, then the origin is just 2308 zeros.
    origin                 = np.zeros(shape=(1, frameFeatureDimensions))

    frameIdToVideoId       = {}
    frameIdToFrameFeature  = {}

    frameCount = 0
    for videoName, features in videoFeatures.items():
        for frameFeature in features:
            frameIdToVideoId[frameCount]      = videoName
            frameIdToFrameFeature[frameCount] = frameFeature

            frameCount += 1


    videoRank      = defaultdict(lambda: 0.0)
    discountFactor = 0.95
    
    # Calculate the distance of all frames to the origin
    frameFeatures = np.asarray(list(frameIdToFrameFeature.values()))
    minDistances  = distance(origin, frameFeatures)
    

    # Every frame must have their "value" calculated
    for _ in tqdm(range(frameCount), desc="Ranking videos", unit="frame"):
        # Select the frame with the largest distance
        selectedFrameId = np.argmax(minDistances)

        # The video ID is used to update the rank of the video from which the selected frame belongs to
        selectedVideoId = frameIdToVideoId[selectedFrameId]

        # The new rank of the video is 1 * the discount factor. Videos that get picked first
        # get larger values.
        videoRank[selectedVideoId] += discountFactor * 1

        # Update the discount factor
        discountFactor = discountFactor * 0.95

        # Update the distance of the selected frame in minDistances (to avoid it getting picked again).
        # I chose to update the value to -1, because this way we guarantee that it will never get picked again
        minDistances[selectedFrameId] = -1


        # Check if the distance from the selected frame to all the remaining frames
        # is lesser than the stored distance of the remaining frames.
        selectedFrameFeature    = frameIdToFrameFeature[selectedFrameId]
        distanceRemainingFrames = distance(selectedFrameFeature, frameFeatures)
        
        # Create a mask for where the distance to the selected frame is lower than the stored distance
        isRemainingFrameCloser = distanceRemainingFrames < minDistances

        # Update min distances according to the mask 
        minDistances[isRemainingFrameCloser] = distanceRemainingFrames[isRemainingFrameCloser]
    

    # Order videos by most informative to least informative
    videoRank = dict(sorted(videoRank.items(), key=lambda x: x[1], reverse=True))

    # Convert from numpy float to regular float. This is because json.dump() can't save numpy floats :(
    videoRank = { k: float(v) for k, v in videoRank.items() }

    return videoRank


'''
points = {
            "Video1": np.asarray(   [
                                        [1,1], [2,2]
                                    ]
                                ),

            "Video2": np.asarray(   [
                                        [1,1], [0.5,0.5]
                                    ]
                                )
}

kCenter(points)'
'''