import unittest
import cv2
import utils
import numpy as np

data = np.array([
                [1.49167023e+02, 1.17518692e+02, 8.89328837e-01],
                [1.33975815e+02, 1.60093048e+02, 8.43872547e-01],
                [1.00968376e+02, 1.60087952e+02, 7.75377691e-01],
                [8.85623550e+01, 1.13326576e+02, 5.92549622e-01],
                [8.16218491e+01, 6.10127029e+01, 8.90402257e-01],
                [1.68349777e+02, 1.60087250e+02, 8.07024300e-01],
                [2.00022202e+02, 1.36755524e+02, 7.83210337e-01],
                [2.20638779e+02, 9.12932129e+01, 8.11998129e-01],
                [1.60168549e+02, 2.66046814e+02, 6.82261884e-01],
                [1.36779709e+02, 2.70205170e+02, 6.77001476e-01],
                [1.05115517e+02, 3.69250580e+02, 8.70005608e-01],
                [5.82596359e+01, 4.44907715e+02, 7.24618673e-01],
                [1.84885788e+02, 2.59180939e+02, 6.28191411e-01],
                [2.02809235e+02, 3.58253876e+02, 8.77936721e-01],
                [2.16549210e+02, 4.44901001e+02, 7.50263155e-01]
                ])

class utils_test(unittest.TestCase):
    def test_import(self):
        try:
            import OpenPose
        except:
            self.fail()

    def test_crop_resize(self):
        new_data = utils.crop_resize_image(data)
        # L2 Norm is performed on data
        self.assertTrue(np.max(new_data) <= 1)
        print(new_data.shape, data.shape)
        self.assertEquals(new_data.shape, (data.shape[0], data.shape[1] - 1))