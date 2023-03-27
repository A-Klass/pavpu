import numpy as np
from typing import List, Optional, TypeVar, Generic, Tuple, Union

from src.utils import _accuracy_map_pooled, _uncertainty_map_pooled, _uncertainty_map

Shape = TypeVar("Shape")
DType = TypeVar("DType")
class Array(np.ndarray, Generic[Shape, DType]):
    """  
    Kudos: https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
    Use this to type-annotate numpy arrays, e.g. 
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """
    pass

def pavpu(target: Array['W, H', 'float'],
          prediction: Array['W, H', 'float'] = None,
          uncertainty: Optional[str] = "aleatoric",
          base_metric: Optional[str] = "accuracy",
          base_metric_threshold: Optional[float] = 0.5, 
          uncertainty_threshold: Optional[float] = 0.4, 
          balanced: Optional[bool] = False,
          padded: Optional[bool] = False,
          sample_weights: Optional[List[float]] = None,
          kernel_size: Optional[int] = 2,
          stride_size: Optional[int] = None,
          average: Optional[str] = "macro",
          manual_unc_map: Optional[np.ndarray] = None,
          manual_hard_labels: Optional[np.ndarray] = None,
          **kwargs) -> float:
    """
    Calculate the Patch Accuracy vs. Patch Uncertainty (PAvPU) score for a given set of 
    predictions and targets.
    
    Implementation for Mukhoti, Gal (2018): Evaluating Bayesian Deep Learning Methods
    for Semantic Segmentation
    https://arxiv.org/pdf/1811.12709.pdf

    
    Note!: 
    -----------
        One batch refers to one Monte Carlo sample. 
        `batch_size` refers to the number of MC samples, not the batch size of the data.
        
        classes should be one-hot encoded (referring to `num_classes` dimension)
        
        num_classes is extracted from the shape of the prediction array to perform
        sanity checks!
          

    Parameters:
    -----------
    prediction : np.ndarray
        An array of shape (batch_size, num_classes, height, width) containing the predicted
        softmax scores for the classes for each input in the batch.
        
        If input shall not be softmax ouputs but rather hard labels, provide inputs
        to `manual_hard_labels` instead.
        
    target : np.ndarray
        Labels of the data. An array of shape (batch_size, num_classes, height, width).
        
    uncertainty : str, optional
        The type of uncertainty to use in the calculation. 
        Can be one of {'total', 'aleatoric', 'epistemic'}. 
        Default is 'aleatoric'.

    base_metric : str, optional
        The base metric to use for calculation of accuracy. 
        Can be one of {'accuracy', 'f1', 'recall', or 'f1_sklm'}. Default is 'accuracy'.
        
    base_metric_threshold : float, optional
        The threshold value to use for the base metric. Default is 0.5.
        
    uncertainty_threshold : float, optional
        The threshold value to use for uncertainty. Default is 0.4.
        
    balanced : bool, optional
        Whether to use balanced accuracy or not. Default is False.

    padded : bool, optional
        Whether to pad the images or not. Default is False.
        
    sample_weights : list of float, optional
        The sample weights for each pixel. Default is None.
        
    kernel_size : int, optional
        The size of the kernel for pooling. Default is 2.
        
    stride_size : int, optional
        The size of the stride for pooling. Default is None, which sets it to kernel_size.

    average : str, optional
        The type of averaging to use. Which type is available depends on the base metric.
        
    manual_unc_map : np.ndarray, optional
        An array containing the uncertainty map for the inputs. Default is None.
        
    manual_hard_labels : np.ndarray, optional
        An array containing the hard labels for the inputs instead of softmax outputs.
        Note that pavpu still expects one-hot encoded labels, hence:
        (batch_size, num_classes, height, width) = (1, num_classes, height, width)
        Default is None.

    Returns:
    --------
    float pavpu_score, a_given_c, u_given_i
        PAvPU score for 1 prediction, mean p(a|c), mean p(uncertain|inaccurate)
    """
    # Step size
    # ----------------
    if stride_size is None:
        stride_size = kernel_size

    # Error message for input shapes
    # ----------------
    err_msg_shape = '; should be of format ((batch_size, num_classes, height, width))' + \
    "with batch_size = number of Monte Carlo samples/predictions and num_classes " + \
    "= number of one-hot encoded classes." + \
    "If necessary, use something like this: " + \
    "Given that you have 3 classes and 1 prediction of shape (512,512), apply:" + \
    "np.expand_dims(get_one_hot(labels), axis = 0),axes=(0,3,1,2))"
    
    # Predictions
    # ----------------
    if manual_hard_labels is not None: 
        # assume that if this is provided, it takes precedence over prediction
        hard_labels_sampled_outputs = manual_hard_labels
        assert len(hard_labels_sampled_outputs.shape) == 4
    elif prediction is None:
        raise ValueError("Either prediction or manual_hard_labels must be provided.")
    else:
        # reverse one-hot encoding and make hard label predictions out of outputs
        # 1. average over softmax predictions for single classes 
        # (meanwhile collapsing axis 0, which is MC samples)
        # 2. now first axis (previously second) is onehot encodings -> 
        # take argmax to collapse axis and get class prediction
        assert len(prediction.shape) == 4, f'prediction.shape: {prediction.shape}' + err_msg_shape
        hard_labels_sampled_outputs = np.argmax(np.mean(prediction, axis = 0), axis = 0) 

    # Target
    # ----------------
    assert len(target.shape) == 4, f'target.shape: {target.shape}' + err_msg_shape
    # target[0] == target[1] since for multiple MC predictions the ground truth is the same
    num_classes = target.shape[1]
    target =  np.argmax(np.array(target), axis = 1)[0]

    # Uncertainty map
    # ----------------
    if manual_unc_map is not None:
        assert len(manual_unc_map.shape) == 2, f'manual_unc_map.shape: {manual_unc_map.shape}' + \
            "manual_unc_map.shape should be something like (512,512)"
        uncertainty_map = manual_unc_map 
    else:
        uncertainty_map = _uncertainty_map(uncertainty_type = uncertainty,
                                           prediction = prediction)
        
    assert uncertainty_map.shape[0] // stride_size >=2, f'stride_size {stride_size} must fit into uncertainty_map.shape[0] {uncertainty_map.shape[0]}'
    assert uncertainty_map.shape[1] // stride_size >=2, f'stride_size {stride_size} must fit into uncertainty_map.shape[0] {uncertainty_map.shape[1]}'
    
    assert uncertainty_map.shape[1] % stride_size == int(0), f"necessary to fulfill uncertainty_map.shape[1] = {uncertainty_map.shape[1]} modulo stride_size = {stride_size} == int(0)"

    accuracy_map_pooled = _accuracy_map_pooled(target = target,
                                               hard_labels_sampled_outputs = hard_labels_sampled_outputs,
                                               balanced = balanced,
                                               stride_size = stride_size,
                                               kernel_size = kernel_size,
                                               base_metric_threshold = base_metric_threshold,
                                               base_metric = base_metric,
                                               sample_weights = sample_weights,
                                               padded = padded,
                                               average = average,
                                               **kwargs) 

    uncertainty_map_pooled = _uncertainty_map_pooled(target = target, # required for sample_weights,
                                                     uncertainty_map = uncertainty_map,
                                                     balanced = balanced,
                                                     stride_size = stride_size,
                                                     kernel_size = kernel_size,
                                                     threshold = uncertainty_threshold,
                                                     sample_weights = sample_weights,
                                                     num_classes = num_classes,
                                                     padded = padded)

    assert np.array_equal(uncertainty_map_pooled.shape, accuracy_map_pooled.shape)

    n_accurate_certain     = np.sum(np.logical_and(accuracy_map_pooled,                  np.logical_not(uncertainty_map_pooled)))
    n_inaccurate_certain   = np.sum(np.logical_and(np.logical_not(accuracy_map_pooled),  np.logical_not(uncertainty_map_pooled)))
    n_accurate_uncertain   = np.sum(np.logical_and(accuracy_map_pooled                 , uncertainty_map_pooled))
    n_inaccurate_uncertain = np.sum(np.logical_and(np.logical_not(accuracy_map_pooled) , uncertainty_map_pooled))

    try:
        if n_accurate_certain == 0.0:
            a_given_c = 0
        else:
            a_given_c = n_accurate_certain / (n_accurate_certain + n_inaccurate_certain)
    except ZeroDivisionError:
        a_given_c = 0.0

    try:
        if n_inaccurate_uncertain == 0.0:
            u_given_i = 0.0
        else:
            u_given_i = n_inaccurate_uncertain / (n_inaccurate_certain + n_inaccurate_uncertain)
    except ZeroDivisionError:
        u_given_i = 0.0

    pavpu_score = (n_accurate_certain + n_inaccurate_uncertain) / (n_accurate_certain + n_accurate_uncertain + n_inaccurate_certain + n_inaccurate_uncertain)

    return pavpu_score, a_given_c, u_given_i
