import numpy as np

from tqdm import tqdm


def distance(vector1, vector2):
    """
    Calculates the eucledian distance between two N-dimensional vectors.

    Args:
        vector1 (numpy.typing.ArrayLike): An N-dimensional vector
        vector2 (numpy.typing.ArrayLike): An N-dimensional vector

    Returns:
        np.typing.ArrayLike: The eucledian distance between the two vectors. 
    """
    return np.sqrt(np.sum((vector1 - vector2) ** 2, axis=1))



def kCenter(unlabeledFeatures, labeledFeatures) -> dict:
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
        unlabeledFeatures (dict): A dictionary where the key is the video Id, and the value are the features for that video.

    Returns:
        dict: A dict where the key is the video Id, and the value is the ranking of said video.
    """

    # The videoRank is a dict where each video has a starting rank of 0, and it increases as
    # frames belonging to said video get selected.
    videoRank              = {}

    # Since each video has multiple frames and KCenter will select FRAMES instead of VIDEOS,
    # we need to know which video each frame belongs to.
    frameIdToVideoId       = {}

    # And also need a way to know what are the features of each video.
    frameIdToFrameFeature  = {}

    # We can add a pseudoID to each frame. Starting at 0 and incrementing iteratively,
    # each frame will get a unique ID.
    frameId = 0
    for videoName, features in unlabeledFeatures.items():
        videoRank[videoName] = 0.0

        for frameFeature in features:
            frameIdToVideoId[frameId]      = videoName
            frameIdToFrameFeature[frameId] = frameFeature

            frameId += 1


    # These are all the frame-wise features bundled up in an array of shape
    # (numberOfFrames, 1024)
    unlabeledFrameFeatures = np.asarray(list(frameIdToFrameFeature.values()))
    
    
    aux = []
    for _, features in labeledFeatures.items():
        for frameFeature in features:
            aux.append(frameFeature)

    labeledFeatures = np.asarray(aux)

    # Calculates the distance from the first labeled frame to every unlabeled frame as a baseline
    minDistances  = distance(labeledFeatures[0], unlabeledFrameFeatures)
    
    # For every labeled frame
    for frameFeature in tqdm(labeledFeatures, unit="frames", desc="Calculating initial distance to labeled set"):
        # Calculate the distance of the current labeled frame to every unlabeled frame
        d = distance(frameFeature, unlabeledFrameFeatures)

        # Update the mindistances where the current distances are smaller than the stored ones
        maskUpdate = d < minDistances
        minDistances[maskUpdate] = d[maskUpdate]
        
    

    # The discount factor influences how much later-selected videos affect the ranking
    baseDiscountFactor = 0.95
    discountFactor     = baseDiscountFactor
    # The discount factor will only be updated every 'discountFactorUpdateFrequency' iterations
    discountFactorUpdateFrequency = 550
    # Because the discount factor can get really low, when it gets below 'discountFactorStop'
    # we early stop the iterations
    discountFactorStop = 0.01
    
    # With the informations above, it is possible to calculate exactly how many iterations there will be
    numIterations = int(np.emath.logn(n=discountFactor, x=discountFactorStop) ) * discountFactorUpdateFrequency

    print(f"Selecting {numIterations} out of {frameId} frames ({100 * numIterations / frameId:.2f}%)")

    # Every frame must have their "value" calculated
    for i in tqdm(range(numIterations), desc="Ranking videos", unit="frame"):

        # TODO Descobrir porque esses vídeos longos não estão sendo selecionados
        #print(videoRank["09July_2009_Thursday_tagesschau_default-3"])
        #print(videoRank["23September_2009_Wednesday_tagesschau_default-6"])
        #print(videoRank["25August_2009_Tuesday_heute_default-5"])
        #print(videoRank["12July_2009_Sunday_tagesschau_default-14"])
        #print(videoRank["27August_2009_Thursday_tagesschau_default-9"])
        #print(videoRank["25August_2009_Tuesday_tagesschau_default-5"])
        #print(videoRank["09August_2009_Sunday_tagesschau_default-15"])
        #print(videoRank["23July_2009_Thursday_tagesschau_default-6"])
        #print(videoRank["05May_2010_Wednesday_tagesschau_default-9"])


        # Select the frame with the largest distance
        selectedFrameId = np.argmax(minDistances)

        # The video ID is used to update the rank of the video from which the selected frame belongs to
        selectedVideoId = frameIdToVideoId[selectedFrameId]

        # The new rank of the video is 1 * the discount factor. Videos that get picked first
        # get larger values.
        videoRank[selectedVideoId] += discountFactor
        print(f"Added {discountFactor} to {selectedVideoId}")

        # Update the discount factor every 'discountFactorUpdateFrequency' epochs
        if ( (i+1) % discountFactorUpdateFrequency) == 0:
            discountFactor = discountFactor * baseDiscountFactor


        # Update the distance of the selected frame in minDistances (to avoid it getting picked again).
        # I chose to update the value to -1, because this way we guarantee that it will never get picked again
        minDistances[selectedFrameId] = -1


        # Check if the distance from the selected frame to all the remaining frames
        # is lesser than the stored distance of the remaining frames.
        selectedFrameFeature    = frameIdToFrameFeature[selectedFrameId]
        distanceRemainingFrames = distance(selectedFrameFeature, unlabeledFrameFeatures)
        
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
