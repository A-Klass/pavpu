import numpy as np
import unittest

class BaseTestCases(unittest.TestCase):
    """
    Provide fixed examples of sampled outputs + target
    1) Exact replica of the original paper
    2) Non-random example of an output with shape (MC samples, #classes, width, height)
    Makes it easier to inherit for different test classes
    """
    def __init__(self, method_name: str = "testcases") -> None:
        super().__init__(method_name)
        # ------------------------------------------------------------------------
        # Paper

        # Labels, predictions and uncertainty map are fixed and taken from 
        # the graphic in the paper.
        # Note that pavpu() expects one-hot encoded labels and predictions
        self.labels_paper = np.array([[1,2,5,7],
                                [6,4,3,3],
                                [10,9,5,0],
                                [8,6,4,4]])
        self.predictions_paper = np.array([[1,2,4,7],
                                    [5,6,3,3],
                                    [10,9,4,0],
                                    [8,7,3,4]])
        self.unc_map_paper = np.array([[0.1, 0.3, 0.6, 0.3],
                                [0.7, 0.6, 0.2, 0.1],
                                [0.2, 0.4, 0.5, 0.3],
                                [0.1,0.7, 0.6, 0.2]])
        self.bin_acc_map_paper = np.array([[0, 1, 1, 0]])
        self.bin_unc_map_paper = np.array([[1, 0, 0, 0]])
        self.n_bin_map = self.bin_acc_map_paper.shape[1]
        # ------------------------------------------------------------------------
        # Custom shape: (MC samples, #classes, width, height)

        # Create test inputs and targets via a non-random example of
        # sampled_outputs with shape (2, 3, 4, 4) with values that sum
        # up to 1 in the second dimension
        # (2, 3, 4, 4) = (MC samples/batch_size, #classes, width, height)
        self.sampled_outputs_custom_shape = np.array([
            [
                # Monte Carlo sample 1
                [
                    # Class 1
                    [0.2, 0.5, 0.8, 0.0],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.0, 0.6, 0.2, 0.2],
                    [0.0, 0.0, 0.07, 0.0]
                    ],
                # Class 2
                [
                    [0.3, 0.4, 0.2, 0.1],
                    [0.2, 0.1, 0.1, 0.2],
                    [0.1, 0.3, 0.3, 0.3],
                    [0.01, 0.05, 0.93, 0.02]
                ],
                # Class 3
                [
                    [0.5, 0.1, 0.0, 0.9],
                    [0.7, 0.2, 0.8, 0.7],
                    [0.9, 0.1, 0.5, 0.5],
                    [0.99, 0.95, 0.0, 0.98]
                ]
            ],
            [
                # Monte Carlo sample 2
                [
                    # Class 1
                    [0.3, 0.4, 0.2, 0.1],
                    [0.2, 0.1, 0.1, 0.2],
                    [0.1, 0.3, 0.3, 0.3],
                    [0.15, 0.15, 0.4, 0.3]
                ],
                    # Class 2
                [
                    [0.2, 0.5, 0.8, 0.0],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.0, 0.6, 0.2, 0.2],
                    [0.15, 0.35, 0.3, 0.2]
                ],
                    # Class 3
                [
                    [0.5, 0.1, 0.0, 0.9],
                    [0.7, 0.2, 0.8, 0.7],
                    [0.9, 0.1, 0.5, 0.5],
                    [0.7, 0.5, 0.3, 0.5]
                ]
            ]
        ], dtype = np.float64)

        self.hard_labels_sampled_outputs_custom_shape = np.argmax(
            np.mean(
                self.sampled_outputs_custom_shape,
                axis = 0,
                keepdims = True),
            axis = 0,
            keepdims = True)

        target_custom_shape = np.array([
            [# Class 1
                [1,1,0,1],
                [1,0,0,1],
                [1,1,0,1],
                [0,0,0,1]
            ],
            [# Class 2
                [0,0,1,0],
                [0,1,1,0],
                [0,0,0,0],
                [1,1,0,0]
             ],
            [# Class 3
                [0,0,0,0],
                [0,0,0,0],
                [0,0,1,0],
                [0,0,1,0]
             ]
            ])
        # multiclass single-label target:
        # same target for both MC predictions
        self.target_custom_shape = np.array([target_custom_shape, target_custom_shape])
