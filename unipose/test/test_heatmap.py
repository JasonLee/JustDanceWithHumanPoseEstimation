import unittest
from dataloader import MPIIDataset
from utils import gaussian_heatmap

class Heatmap_Test(unittest.TestCase):
    def test_gaussian(self):
        centre = 2
        heatmap = gaussian_heatmap((centre, centre), 5, 1)

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

if __name__ == '__main__':
    unittest.main()