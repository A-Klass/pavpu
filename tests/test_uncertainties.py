import unittest
import numpy as np

from src.uncertainties import total_unc, aleatoric_unc, epistemic_unc, entropy
from tests.base_testcases import BaseTestCases

class TestUncFunctionsCustomShape(BaseTestCases):
    """
    Class for testing the uncertainty functions
    """
    def __init__(self, method_name: str = "runTestUnc") -> None:
        super().__init__(method_name)
        
        self.sampled_outputs = self.sampled_outputs_custom_shape
        self.target = self.target_custom_shape
        
        self.total_unc_output = total_unc(self.sampled_outputs, self.target)
        self.alea_unc_output = aleatoric_unc(self.sampled_outputs, self.target)
        self.epis_unc_output = epistemic_unc(self.sampled_outputs, self.target)
        
        self.max_unc = np.log2(self.target.shape[1])
        self.min_unc = 0.0
        
    def test_custom_shape(self):
        self.assertEqual(self.sampled_outputs.shape, (2, 3, 4, 4))
        self.assertEqual(self.target.shape, (2, 3, 4, 4))
        
    def test_sanity_checks(self):
        # check that the sampled softmax outputs sum up to 1. note that we're
        # comparing floats -> use np.isclose
        self.assertTrue(np.isclose(np.sum(self.sampled_outputs, axis = 1), 1.0, atol = 1e-64).all())
        self.assertTrue(np.isclose(np.sum(self.target, axis = 1), int(1), atol = 1e-64).all())

        self.assertTrue(np.all(np.sum(self.target, axis = 1) == np.ones((2,4,4))))

    def test_entropy(self):
        self.assertFalse(np.isnan(entropy(self.sampled_outputs)).all())
        self.assertTupleEqual(entropy(self.sampled_outputs).shape,
                              (self.sampled_outputs.shape[0],
                               self.sampled_outputs.shape[2],
                               self.sampled_outputs.shape[3]) )

    def test_output_shape(self):
        self.assertEqual(self.total_unc_output.shape, self.target.shape[2:])
        self.assertEqual(self.alea_unc_output.shape, self.target.shape[2:])
        self.assertEqual(self.epis_unc_output.shape, self.target.shape[2:])

    def test_output_values(self):
        self.assertTrue(np.all(self.total_unc_output >= self.min_unc))
        self.assertTrue(np.all(self.alea_unc_output >= self.min_unc))
        self.assertTrue(np.all(self.epis_unc_output >= self.min_unc))

        self.assertTrue(np.all(self.total_unc_output <= self.max_unc))
        self.assertTrue(np.all(self.alea_unc_output <= self.max_unc))
        self.assertTrue(np.all(self.epis_unc_output <= self.max_unc))

    def test_nan_values(self):
        self.assertFalse(np.any(np.isnan(self.total_unc_output)))
        self.assertFalse(np.any(np.isnan(self.alea_unc_output)))
        self.assertFalse(np.any(np.isnan(self.epis_unc_output)))

    def test_total_unc(self):
        # Calculate expected output
        mean_output = np.mean(self.sampled_outputs, axis=0)
        expected_output = entropy(mean_output, axis = 0)

        # Test overall total_uncertainty calculation
        output = total_unc(self.sampled_outputs)
        self.assertTrue(np.allclose(output, expected_output))

    def test_aleatoric_unc(self):
        # TODO: Implement test
        # Test the "none" average
        # Test the "weighted_by_support" average
        pass
    def test_epistemic_unc(self):
        # TODO: Implement test
        # Test the default micro average
        pass

if __name__ == '__main__':
    unittest.main()
    