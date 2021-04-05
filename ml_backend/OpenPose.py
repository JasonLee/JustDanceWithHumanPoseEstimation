# From Python
# It requires OpenCV installed for Python
from .PoseEstimation import PoseEstimation
import sys
import cv2
import os
# import DTW
import numpy as np
from .utils import limb_breakdown

class OpenPose(PoseEstimation):
    def __init__(self):
        try:
            # Import Openpose (Windows)
            # dir_path = os.path.dirname(os.path.realpath(__file__))
            dir_path = "C:/Users/jlee1/Desktop/Projects/JustDanceWithHumanPoseEstimation/ml_backend/"
            try:
                global op
                # Windows Import
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + './openpose/python/openpose/Release')
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/openpose/x64/Release;' +  dir_path + '/openpose/bin;'
                import pyopenpose as op
            except ImportError as e:
                print('Error: OpenPose library could not be found')
                raise e

            # Custom Params (refer to include/openpose/flags.hpp for more parameters)
            params = dict()
            params["model_folder"] = dir_path + "/openpose/models/"

            # Starting OpenPose
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(params)
            self.opWrapper.start()
            self.datum = op.Datum()

        except Exception as e:
            print(e)
            sys.exit(-1)
    
    def _process_image(self, image):
        try:
            # Process Image
            self.datum.cvInputData = image
            self.opWrapper.emplaceAndPop([self.datum])
            return self.datum

        except Exception as e:
            print(e)
            return -1

    def process_image_keypoints(self, image):
        datum = self._process_image(image)

        if datum != -1:
            return np.copy(datum.poseKeypoints)
        else:
            print("Error: could not process image")

    def get_cv_image(self, image):
        datum = self._process_image(image)

        if datum != -1:
            return np.copy(datum.cvOutputData)
        else:
            print("Error: could not process image")

    def get_joint_mapping(self):
        arr = [0, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13]
        return arr

if __name__ == '__main__':
    op = OpenPose()

    imageToProcess = cv2.imread("../images/image2.png")
    data = op.process_image_keypoints(imageToProcess)
    imagecv = op.get_cv_image(imageToProcess)

    imageToProcess = cv2.imread("../images/image3.png")
    data2 = op.process_image_keypoints(imageToProcess)
    imagecv2 = op.get_cv_image(imageToProcess)
    
    limb_breakdown(data[0, :15], data2[0, :15])


    cv2.imshow("similar1", imagecv)
    cv2.imshow("image3", imagecv2)
    cv2.waitKey(0)