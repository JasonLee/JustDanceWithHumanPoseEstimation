## Dynamic Time Warping
import numpy as np
import math

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
    return dtw_matrix.flatten()[-1]


def dist_func(arr1, arr2):
    if len(arr1) != len(arr2):
        print("Lengths of arrays in distance function are not equal")
        return -1

    arr_len = len(arr1)
    cosine_sim_arr = []

    for i in range(arr_len):
        # if keypoint hasn't been recognised
        if not all(i == 0.0 for i in arr1[i]) or not all(i == 0.0 for i in arr2[i]):
            pass

        cosine_sim_arr.append(cosine_sim_vector(arr1[i], arr2[i]))

    return round(np.mean(cosine_sim_arr))

def cosine_sim_vector(vec1, vec2):
    x1 = vec1[0]
    y1 = vec1[1]
    x2 = vec2[0]
    y2 = vec2[1]

    mag_A = math.sqrt(x1**2 + y1**2) 
    mag_B = math.sqrt(x2**2 + y2**2)

    mag_AB = mag_A * mag_B
    dot_product = (x1 * x2) + (y1 * y2)

    cos_theta = dot_product / mag_AB

    # Floating point error so result is rounded to 3.d.p
    return round(cos_theta, 3)

# # Testing
# a1 = [ [[1,1], [1,1], [1,1]], [[1,1], [1,1], [1,1]] ,[[1,1], [2,1], [3,1]] ]
# a2 = [ [[1,1], [2,1], [3,1]] ]
# print(dtw(a1, a2))
