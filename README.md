# PAvPU  (Patch Accuracy vs. Patch Uncertainty) Implementation
This repository contains the implementation of the Patch Accuracy vs. Patch Uncertainty (PAvPU) metric from the paper: https://arxiv.org/pdf/1811.12709.pdf.
The metric is implemented in NumPy with a foucs on optimizing w.r.t. speed.

# Novel Extensions
- Padding (not in the original paper)
- Alternative metrics such as F1, Recall (not in the original paper) to replace Accuracy.
- Weighting scheme to account for class imbalance (not in the original paper).

# Installation
```
cd pavpu
python setup.py install
```
Soon to be released on PyPI.
# Testing
Testing was done on 
- the example illustration from the paper and 
- on a custom example which represents 2 Monte Carlo samples with 3 classes and 4x4 pixels (2, 3, 4, 4).


# Examples
```
mc_predictions = np.array([
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

hard_labels = np.argmax(
                np.mean(
                    self.sampled_outputs_custom_shape,
                    axis = 0,
                    keepdims = True),
                    axis = 0,
                keepdims = True)

lables =  = np.array([
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

pavpuscore, a_given_c, u_given_i = pavpu(prediction = hard_labels,
                                        target = labels,
                                        uncertainty_threshold = 0.3,
                                        base_metric_threshold = 0.1,
                                        base_metric = "accuracy")
```
# Dependencies
tbd