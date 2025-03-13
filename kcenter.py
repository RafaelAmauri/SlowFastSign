import numpy as np
from collections import defaultdict


def distance(vector1, vector2) -> float:
    """
    Calculated the eucledian distance between two N-dimensional vectors.

    Args:
        vector1 (numpy.typing.ArrayLike): An N-dimensional vector
        vector2 (numpy.typing.ArrayLike): An N-dimensional vector

    Returns:
        float: The eucledian distance
    """
    if vector1.shape != vector2.shape:
        raise ValueError(f"Vectors must have the same shape! Detected shapes: V1 = {vector1.shape}, V2 = {vector2.shape}")

    return np.sqrt(np.sum((vector1 - vector2) ** 2))



def kCenter(points, nCenters: int) -> dict:
    """
    Finds K different centers in an N-dimensional space.

    This is an implementation of the Coreset Active Learning algorithm.
    
    
    Coreset works as a farthest-first selection. We start with list of selected points that contains a single point.
    We then add the point that is the farthest to this one to the list of selected points.
    Next, we add the point that is the farthest to the two that exist on the list. We iteratively do this
    until we have 'nCenters' points.
    

    Args:
        points (dict) : A dict of the points. They must all have the same dimensions.
        nCenters (int): The number of centers. Must be lower than the number of points.

    Returns:
        dict: _description_
    """
    if nCenters > len(points):
        raise ValueError(f"The number of centers must be lower than the number of points. N points: {len(points)}, N centers: {nCenters}")

       
    # Add the first point as the starting point for Coreset
    firstPointName, firstPoint = list(points.items())[0]
    selectedCenters = {
                        firstPointName: firstPoint
                    }
    
    
    lowestDistanceFromCenters = defaultdict(lambda: [float('inf'), -1])
    
    for currentCenterIdx in range(nCenters):
        _, center = selectedCenters[currentCenterIdx]
        
        for candidateName, candidate in points.items():
            d = distance(center, candidate)

            if d < lowestDistanceFromCenters[candidateName][0]:
                lowestDistanceFromCenters[candidateName] = [d, candidate]
    
    
        # Sort the distances from each point to the closest center
        lowestDistanceFromCenters  = sorted(lowestDistanceFromCenters.items(), key=lambda x: x[1][0], reverse=True)
        
        closestCandidateName, closestCandidate = points(lowestDistanceFromCenters[0][0])

        # Add the farthest point to the list of centers and delete it from the list of points
        selectedCenters[closestCandidateName] = closestCandidate
        del points[closestCandidateName]


    return selectedCenters


points = {
            "1": np.asarray([1,   1,  1,  1]),
            "2": np.asarray([3,   3,  3,  3]),
            "3": np.asarray([10, 10, 10, 10]),
            "4": np.asarray([12, 12, 12, 12]),
            "5": np.asarray([40, 40, 40, 40])
}

print(kCenter(points, 3))