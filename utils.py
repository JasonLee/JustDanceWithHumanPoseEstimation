import math
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

THRESHOLD_PERFECT = 0.8
THRESHOLD_GOOD = 0.7
THRESHOLD_BAD = 0.6
SCORE_PERFECT = 5
SCORE_GOOD = 3
SCORE_BAD = 1
SCORE_X = 0

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 480

KEYPOINT_NUM = 15

# Many to 1: Large to Small  https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering-in-c-python
COMPARISION_DICT = [0, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]

def L2_norm(arr):
    dist = np.sqrt(np.sum(np.square(arr)))
    arr = arr / dist

    print(arr)
    return arr

def convert_to_vector(data):
    # Delete row of 0s 
    # data = data[~np.all(data == 0, axis=1)]
    arr = np.delete(data, -1, 1)
    return arr.flatten()

def get_bounding_box(keypoints):
    min_y = 0
    max_y = 0
    min_x = 0
    max_x = 0

    for i in range(KEYPOINT_NUM):
        if min_x > keypoints[i][0]:
            min_x = keypoints[i][0]
        if max_x <= keypoints[i][0]:
            max_x = keypoints[i][0]

        if min_y > keypoints[i][1]:
            min_y = keypoints[i][1]
        if max_y <= keypoints[i][1]:
            max_y = keypoints[i][1]

    return min_x, min_y, max_x, max_y

def crop_resize_image(input_keypoints):
    input_keypoints = np.delete(input_keypoints, -1, 1)

    # Crop coords
    min_x, min_y, max_x, max_y = get_bounding_box(input_keypoints)
    resize_ratio_x = IMAGE_WIDTH / (max_x - min_x)
    resize_ratio_y = IMAGE_HEIGHT / (max_y - min_y)

    # Keypoints are "cropped" then resized to 480x480
    for i in range(KEYPOINT_NUM):
        input_keypoints[i][0] = (input_keypoints[i][0] - min_x) * resize_ratio_x
        input_keypoints[i][1] = (input_keypoints[i][1] - min_y) * resize_ratio_y

    input_keypoints = L2_norm(input_keypoints)

    return input_keypoints
    

def limb_breakdown(input, target):
    # Remove Confidence
    input = crop_resize_image(input)
    target = crop_resize_image(target)

    results = []
    test = []

    cos = cosine_dist_test(input, target)

    for i in range(KEYPOINT_NUM):
        results.append(result_conversion(cos[i]))
        test.append(cos[i])

    # for i in range(KEYPOINT_NUM):
    #     cos = cosine_dist(input[i], target[i])
    #     results.append(result_conversion(cos))
    #     test.append(cos)

    for i in range(len(test)):
        print(i, test[i], "| score", results[i])
        
    test_plot(input, 1, test)
    test_plot(target, 2, None)
    plt.show()
    print(results)
    print(input)
    print(target)
    
def test_plot(keypoint, subplot, result):
    plt.subplot(1, 2, subplot)
    plt.scatter(*zip(*keypoint))

    for i in range(KEYPOINT_NUM):
        if result:
            plt.text(keypoint[i,0], keypoint[i,1], str(i) + ": " + str(result[i]))
        else:
            plt.text(keypoint[i,0], keypoint[i,1], str(i))

    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])

def cosine_sim_vector(a, b):
    cos_sim = np.dot(a, b) / (norm(a)*norm(b))
    return cos_sim

def cosine_dist_test(a,b):
    results = []
    for i in range(KEYPOINT_NUM):
        if i == 0:
            results.append(0)
            continue

        a1 = np.subtract(a[i], a[COMPARISION_DICT[i]])
        b1 = np.subtract(b[i], b[COMPARISION_DICT[i]])

        res = 1 - np.sqrt((1.0 - cosine_sim_vector(a1, b1)))  
        results.append(res)

    results = np.round(results, 2)
    return results

    
def cosine_dist(a, b):
    return 1 - np.cbrt((1.0 - cosine_sim_vector(a, b)))  

def result_conversion(val):

    if val >= THRESHOLD_PERFECT:
        return SCORE_PERFECT
    elif val >= THRESHOLD_GOOD:
        return SCORE_GOOD
    elif val >= THRESHOLD_BAD:
        return SCORE_BAD
    else:
        return SCORE_X

