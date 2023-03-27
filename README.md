# PAvPU  (Patch Accuracy vs. Patch Uncertainty) Implementation
This repository contains the implementation of the Patch Accuracy vs. Patch Uncertainty (PAvPU) metric from the paper: https://arxiv.org/pdf/1811.12709.pdf.
The metric is implemented in NumPy with a foucs on optimizing w.r.t. speed.

# Extensions:
- Padding (not in the original paper)
- Alternative metrics such as F1, Recall (not in the original paper) to replace Accuracy.
- Weighting scheme to account for class imbalance (not in the original paper).

# Installation
tbd
# Testing
Testing was done on 
- the example illustration from the paper and 
- on a custom example which represents 2 Monte Carlo samples with 3 classes and 4x4 pixels (2, 3, 4, 4).

# Examples
tbd

# Dependencies
tbd