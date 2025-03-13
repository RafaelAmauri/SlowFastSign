import numpy as np
from collections import defaultdict

def distance(vector1, vector2):
    """Calculated the eucledian distance between two N-dimensional vectors.

    Args:
        vector1 (numpy.typing.ArrayLike): An N-dimensional vector
        vector2 (numpy.typing.ArrayLike): An N-dimensional vector

    Returns:
        float: The eucledian distance
    """
    if vector1.shape != vector2.shape:
        raise ValueError(f"Vectors must have the same shape! Detected shapes: V1 = {vector1.shape}, V2 = {vector2.shape}")

    return np.sqrt(np.sum((vector1 - vector2) ** 2))


def kCenter(points, nCenters):
    """_summary_

    Args:
        points (_type_): _description_
    """
    firstPointName = list(points.keys())[3]
    centers = {
                firstPointName : points[firstPointName]
            }
    
    del points[firstPointName]

    for i in range(1, nCenters):
        lowestDistances = defaultdict(lambda: [float('inf'), -1])
        
        for centerName, center in centers.items():
            
            for candidateName, candidate in points.items():
                d = distance(center, candidate)

                if d < lowestDistances[candidateName][0]:
                    lowestDistances[candidateName] = [d, candidate] 
        
        lowestDistances                = sorted(lowestDistances.items(), key=lambda x: x[1], reverse=True)
        centers[lowestDistances[0][0]] = lowestDistances[0][1][1]
        del points[lowestDistances[0][0]]

    return centers

points = {
            "1": np.asarray([1,   1,  1,  1]),
            "2": np.asarray([3,   3,  3,  3]),
            "3": np.asarray([10, 10, 10, 10]),
            "4": np.asarray([12, 12, 12, 12]),
            "5": np.asarray([40, 40, 40, 40])
}

print(kCenter(points, 3))