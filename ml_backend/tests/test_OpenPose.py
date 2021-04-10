import unittest
import OpenPose
import cv2
import numpy as np

IMAGE_PATH = "tests/images/image.png"
DIFF_IMAGE_PATH = "tests/images/diff_image.png"

class OpenPose_test(unittest.TestCase):
    def test_import(self):
        try:
            import OpenPose
        except:
            self.fail()
    
    def test_processing_images(self):
        op = OpenPose.OpenPose()
        imageToProcess = cv2.imread(IMAGE_PATH)
        data = op.process_image_keypoints(imageToProcess)
        self.assertTrue(type(data) is np.ndarray)

    def test_get_cv_image(self):
        op = OpenPose.OpenPose()
        imageToProcess = cv2.imread(IMAGE_PATH)
        data = op.get_cv_image(imageToProcess)

        self.assertTrue(type(data) is np.ndarray)

    def test_get_cv_image(self):
        op = OpenPose.OpenPose()
        mapping = op.get_joint_mapping()

        self.assertTrue(type(mapping) is list)

    def test_repeated_use(self):
        op = OpenPose.OpenPose()
        imageToProcess = cv2.imread(IMAGE_PATH)
        try:
            data = op.get_cv_image(imageToProcess)
            data1 = op.get_cv_image(imageToProcess)
            data2 = op.get_cv_image(imageToProcess)

        except:
            self.fail()

        self.assertTrue(type(data) is np.ndarray)
        self.assertTrue(type(data1) is np.ndarray)
        self.assertTrue(type(data2) is np.ndarray)

    # Makes sure with different images, different results are outputted
    def test_repeated_use_sanity(self):
        op = OpenPose.OpenPose()
        image = cv2.imread(IMAGE_PATH)
        diff_image = cv2.imread(DIFF_IMAGE_PATH)

        try:
            image_joints = op.get_cv_image(image)
            diff_image_joints = op.get_cv_image(diff_image)

        except:
            self.fail()

        self.assertTrue(not np.array_equal(image_joints, diff_image_joints)) 

