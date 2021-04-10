## Dynamic Time Warping
import numpy as np
from numpy.linalg import norm
import math
import utils

## The Output of OpenPose is as folows
# An array of People which contain a pose keypoint array which is contains x y and confidence
# Keypoint ordering is https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering-in-c-python

# arr1 and arr2 is an array of keypoint array(which is an array)
# e.g. [[(1,2,0.9), (1,2,0.9), ...]]
def dtw(arr1, arr2, window_size=3):
    n, m = len(arr1), len(arr2)

    dtw_matrix = np.zeros((n+1, m+1))

    w = max(window_size, abs(n-m)) # adapt window size

    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    

    dtw_matrix[0, 0] = 0
    for i in range(1, n+1):
        for j in range(max(1, i-w) , min(m+1, i+w+1)):
            dtw_matrix[i, j] = 0


    for i in range(1, n+1):
        for j in range(max(1, i-w) , min(m+1, i+w+1)):
            cost = dist_func(arr1[i-1], arr2[j-1])

            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min

    # Get last element which is the similarity
    print (dtw_matrix)
    return dtw_matrix.flatten()[-1]


def dist_func(arr1, arr2):
    if len(arr1) != len(arr2):
        print("Lengths of arrays in distance function are not equal")
        return -1

    arr1 = utils.L2_norm(arr1)
    arr2 = utils.L2_norm(arr2)

    # cosine sim 1 = very similar, 0 not similar
    # For distance we do 1 - cosine_sim 
    dist = np.sqrt(2 * (1.0 - cosine_sim_vector(arr1, arr2)))

    return dist

def cosine_sim_vector(a, b):
    cos_sim = np.dot(a, b) / (norm(a)*norm(b))
    return cos_sim