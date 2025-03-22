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
    
    
    Coreset works as a farthest-first selection. 
    
    The algorithm:
      1. Picks an arbitrary starting point.
      2. For each remaining point, calculates its distance to the nearest selected center.
      3. Selects the point with the maximum minimum distance.
      4. Repeats until k centers are chosen.
    

    Args:
        points (dict) : A dictionary of points, each as a numpy array.
        nCenters (int): The number of centers (must be lower than or equal to the number of points).

    Returns:
        dict: _description_
    """
    if nCenters > len(points):
        raise ValueError(f"The number of centers must be lower than the number of points. N points: {len(points)}, N centers: {nCenters}")

    remainingPoints = points.copy()
    selectedCenters = {}
    
    # Pick an arbitrary starting point (first in the dictionary)
    _, firstPoint = next(iter(remainingPoints.items()))
    
    # Pick an arbitrary starting point (0, 0)
    selectedCenters["origin"] = np.zeros_like(firstPoint)
    firstPoint = selectedCenters["origin"]

    # Initialize minimum distances for remaining points from the first center
    minDistances = {name: distance(firstPoint, i) for name, i in remainingPoints.items()}
    
    # Select centers until we reach the desired number
    while len(selectedCenters) <= nCenters:
        # Find the point with the maximum distance to its nearest center
        nextCenterName, nextCenterDistance = max(minDistances.items(), key=lambda x: x[1])
        nextCenterCoordinates              = remainingPoints[nextCenterName]
        selectedCenters[nextCenterName]    = [nextCenterCoordinates, nextCenterDistance]

        # Remove the selected point from remaining points and min_distances
        del remainingPoints[nextCenterName]
        del minDistances[nextCenterName]

        # Update the minimum distances for the remaining points
        for name, pt in remainingPoints.items():
            d = distance(nextCenterCoordinates, pt)
            if d < minDistances[name]:
                minDistances[name] = d


    del selectedCenters["origin"]
    return selectedCenters

'''
points = {
            "1": np.asarray([1,   1,  1,  1]),
            "2": np.asarray([3,   3,  3,  3]),
            "3": np.asarray([10, 10, 10, 10]),
            "4": np.asarray([12, 12, 12, 12]),
            "5": np.asarray([40, 40, 40, 40])
}

print(kCenter(points, 5))
'''