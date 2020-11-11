# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import argparse
import DTW
import numpy as np
import utils

class OpenPose:
    def __init__(self):
        try:
            # Import Openpose (Windows)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            try:
                global op
                # Windows Import
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + './openpose/python/openpose/Release');
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/openpose/x64/Release;' +  dir_path + '/openpose/bin;'
                import pyopenpose as op
            except ImportError as e:
                print('Error: OpenPose library could not be found')
                raise e

            # Custom Params (refer to include/openpose/flags.hpp for more parameters)
            params = dict()
            params["model_folder"] = "./openpose/models/"

            # Starting OpenPose
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(params)
            self.opWrapper.start()
            self.datum = op.Datum()

        except Exception as e:
            print(e)
            sys.exit(-1)

    # Testing cv2.imread imageToProcess
    def process_image(self, imageToProcess):
        try:
            # Process Image
            self.datum.cvInputData = imageToProcess
            self.opWrapper.emplaceAndPop([self.datum])
        
            return self.datum

        except Exception as e:
            print(e)
            sys.exit(-1)

    def test(self):
        print("Hello")

if __name__ == '__main__':
    op = OpenPose()

    imageToProcess = cv2.imread("images/image2.png")
    datum = op.process_image(imageToProcess)
    data = np.copy(datum.poseKeypoints)
    image1 = np.copy(datum.cvOutputData)

    imageToProcess = cv2.imread("images/image3.png")
    datum = op.process_image(imageToProcess)
    data2 = datum.poseKeypoints
    
    utils.limb_breakdown(data[0, :15], data2[0, :15])


    cv2.imshow("similar1", image1)
    cv2.imshow("image3", datum.cvOutputData)
    cv2.waitKey(0)