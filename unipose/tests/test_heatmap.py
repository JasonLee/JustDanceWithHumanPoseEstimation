import unittest
import numpy as np
from utils import gaussian_heatmap, local_maxima, evaluation_pckh

class Heatmap_Test(unittest.TestCase):
    def test_gaussian(self):
        import matplotlib.pyplot as plt
        centre = 2
        heatmap = gaussian_heatmap((centre, centre), 100, 3)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()
        print(heatmap)

        if heatmap[centre, centre] != 1:
            self.fail()
        
        # Look around centre point. Must be gaussian not 1 or 0
        if heatmap[centre+1, centre] == 1 or heatmap[centre+1, centre] <= 0:
            self.fail()

        if heatmap[centre-1, centre] == 1 or heatmap[centre-1, centre] <= 0:
            self.fail()

        if heatmap[centre, centre+1] == 1 or heatmap[centre, centre+1] <= 0:
            self.fail()

        if heatmap[centre, centre-1] == 1 or heatmap[centre, centre-1] <= 0:
            self.fail()

        self.assertTrue(heatmap[centre+1, centre] > heatmap[centre+2, centre])
        self.assertTrue(heatmap[centre-1, centre] > heatmap[centre-2, centre])
        self.assertTrue(heatmap[centre, centre+1] > heatmap[centre, centre+2])
        self.assertTrue(heatmap[centre, centre+1] > heatmap[centre, centre+2])

    def test_coords(self):
        centres = [[2, 3], [7,6], [1, 11], [4,4], [0,0]]
        flipped_centres = np.flip(centres, 0)

        size = 12
        heatmaps = np.zeros((2, len(centres), size, size), dtype=int)

        for i in range(5):
            heatmaps[0,i] = gaussian_heatmap(centres[i], size, 1)
            heatmaps[1,i] = gaussian_heatmap(flipped_centres[i], size, 1)

        coords = local_maxima(heatmaps)
        
        for i in range(5):
            self.assertTrue((centres[i] == coords[0][i]).all())
        for i in range(5):
            self.assertTrue((flipped_centres[i] == coords[1][i]).all())

    # @unittest.skip("reason for skipping")
    def test_eval(self):
        data = [[2, 3], [7,6], [1, 11], [4,4], [0,0], [1,4], [6,3], [2,2], [4,4]]
        centres = np.zeros( (2, len(data), 2), dtype=int)
        centres[0,:] = data
        centres[1,:] = np.flip(data, 0)
        size = 12
        heatmaps = np.zeros((2, len(data), size, size), dtype=int)

        for i in range(len(data)):
            heatmaps[0,i] = gaussian_heatmap(centres[0,i], size, 1)
            heatmaps[1,i] = gaussian_heatmap(centres[1,i], size, 1)

        mPCKH, mDist = evaluation_pckh(heatmaps, centres)
        self.assertEqual(mPCKH, 100)







if __name__ == '__main__':
    unittest.main()