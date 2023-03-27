import numpy as np
import unittest

from src.pavpu import pavpu
from base_testcases import BaseTestCases

def _get_one_hot(targets, nb_classes: int = None):
    """
    Transforms a vector of integers into an array of one-hot encoded vectors
    """
    if nb_classes is None:
        nb_classes = len(np.unique(targets))
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

class TestPavPUPaper(BaseTestCases):
    """
    Checks the pavpu function against the example in the original paper (page 5) 
    """
    def __init__(self, method_name: str = "TestPavPUPaper"):
        """
        Labels, predictions and uncertainty map are fixed and taken from 
        the graphic in the paper.
        Note that pavpu() expects one-hot encoded labels and predictions
        """
        super().__init__(method_name)

        self.labels = self.labels_paper 
        self.predictions = self.predictions_paper
        self.unc_map = self.unc_map_paper
        self.bin_acc_map = self.bin_acc_map_paper
        self.bin_unc_map = self.bin_unc_map_paper

        self.labels_one_hot = np.transpose(
            np.expand_dims(_get_one_hot(self.labels),
                           axis = 0),
            axes=(0,3,1,2)
            )
        self.predictions_one_hot = np.transpose(
            np.expand_dims(
                _get_one_hot(self.predictions),
                axis = 0),
            axes=(0,3,1,2)
            )
        
    def test_pavpufunction(self):
        """
        Tests the pavpu function against the example in the paper with fixed uncertainty map
        """
        expected_pavpu_score = (self.bin_acc_map.sum() + self.bin_unc_map.sum()) / self.n_bin_map
        expected_a_given_c = self.bin_acc_map.sum() / np.logical_not(self.bin_unc_map).sum()
        expected_u_given_i = self.bin_unc_map.sum() / self.bin_acc_map.sum()

        self.assertAlmostEqual(expected_a_given_c, 2/3)
        self.assertAlmostEqual(expected_u_given_i, 0.5)
        self.assertAlmostEqual(expected_pavpu_score, 3/4)

        pavpuscore, a_given_c, u_given_i = pavpu(prediction = self.predictions_one_hot,
                                                 target = self.labels_one_hot,
                                                 manual_unc_map = self.unc_map,
                                                 uncertainty_threshold = 0.4,
                                                 base_metric_threshold = 0.5,
                                                 base_metric = "accuracy")

        self.assertAlmostEqual(pavpuscore, expected_pavpu_score, places=10)
        self.assertAlmostEqual(a_given_c, expected_a_given_c, places=10)
        self.assertAlmostEqual(u_given_i, expected_u_given_i, places=10)
        
        self.assertWarns(UserWarning,
                          pavpu,
                          self.labels_one_hot,
                          self.predictions_one_hot,
                          "aleatoric",
                          "f1",         
                          0.1,
                          0.0001)

        self.assertRaises(ValueError,
                          pavpu,
                          self.labels_one_hot,
                          self.predictions_one_hot,
                          "aleatoric",
                          "accuracy",
                          0.1,
                          -0.01)

class TestPavPUCustomShapes(BaseTestCases):
    def __init__(self, method_name: str = "TestCustomShapes") -> None:
        super().__init__(method_name)
        self.sampled_outputs = self.sampled_outputs_custom_shape
        self.target = self.target_custom_shape
                
        self.max_unc = np.log2(self.target.shape[1])
        self.min_unc = 0.0

    def test_pavpufunction(self):
        """
        Tests the pavpu function against the fixed 2nd outputs/target combo
        from the BaseTestCases class
        """
        pavpuscore, a_given_c, u_given_i = pavpu(prediction = self.sampled_outputs,
                                                 target = self.target,
                                                 uncertainty_threshold = 0.3,
                                                 base_metric_threshold = 0.1,
                                                 base_metric = "accuracy")
        self.assertGreaterEqual(pavpuscore, 0.0)
        self.assertGreaterEqual(a_given_c, 0.0)
        self.assertGreaterEqual(u_given_i, 0.0)
        
        self.assertLessEqual(pavpuscore, 1.0)
        self.assertLessEqual(a_given_c, 1.0)
        self.assertLessEqual(u_given_i, 1.0)
        
        # return must be scalar
        self.assertIsInstance(pavpuscore, (float, int))
        self.assertIsInstance(a_given_c, (float, int))
        self.assertIsInstance(u_given_i, (float, int))
        
        self.assertWarns(UserWarning,
                          pavpu,
                          self.target,
                          self.sampled_outputs,
                          "aleatoric",
                          "f1",
                          0.1,
                          0.0001)
        
        self.assertRaises(ValueError,
                          pavpu, 
                         self.target, 
                         self.sampled_outputs,
                         "aleatoric", 
                         "f1",
                         0.1,
                         np.log2(3)+0.1)
        
if __name__ == '__main__':
    unittest.main()
    