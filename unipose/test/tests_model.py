import unittest
from dataloader import MPIIDataset
from unipose import UniPose
import torch

class Model_Test(unittest.TestCase):
    def test_unipose(self):
        model = UniPose()
        model.eval()
        input = torch.rand(1, 3, 368, 368)
        output = model(input)
        
        if list(output.shape) != [1, 16, 368, 368]:
            self.fail("Output does not match")


if __name__ == '__main__':
    unittest.main()