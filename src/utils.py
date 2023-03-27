from math import isclose
import numpy as np
from skimage.util import view_as_windows
from sklearn import metrics as sklm
from typing import Optional
import warnings

from src.uncertainties import aleatoric_unc, epistemic_unc, total_unc

def _accuracy_map_pooled(target, 
                         hard_labels_sampled_outputs, 
                         balanced: bool = False,
                         stride_size: int = 1,
                         kernel_size: int = 1,
                         base_metric_threshold: float = 0.49,
                         base_metric = "accuracy",
                         padded = False,
                         sample_weights: Optional[np.ndarray] = None,
                         average: str = "weighted",
                         **kwargs):
    """
    Returns a pooled accuracy map. Optionally, make balanced averages
    (i.e.: weighted by 1/pixels_per_class) before thresholding.
    
    Args:
    ----------
        target: (np.ndarray) The true labels.
        hard_labels_sampled_outputs: (np.ndarray) The predicted labels.
        balanced: (bool) Whether to weight each class by 1/pixels_per_class.
        stride_size: (int) The stride size for pooling.
        kernel_size: (int) The kernel size for pooling.
        base_metric_threshold: (float) The threshold for the base metric.
        base_metric: (str) The base metric for thresholding.
        padded: (bool) Whether to pad the image before pooling.
        sample_weights: (np.ndarray) The weights for each sample.
        average: (str) The type of averaging to perform.

    Returns:
    ----------
        np.ndarray: A pooled accuracy map.
        
    Raises:
    ----------
    AssertionError: If the target is not quadratical.
    ValueError: 
        -If the averaging method (average:str) is not supported.
        -if threshold is beyond the range [0,1].
        
    Warns: 
    ----------
    If threshold is close to or beyond an edge case.
    """
    # Sanity checks and warnings for edge case thresholds:
    if base_metric_threshold > 1 or base_metric_threshold < 0:
        # hard error since unclear how negative values etc will affect the function's calculations
        raise ValueError("base_metric_threshold is outside of the range [0,1].")
    if isclose(base_metric_threshold, 1.0, abs_tol = 0.01) or isclose(base_metric_threshold, 0.0, abs_tol = 0.01):
        raise UserWarning(f"base_metric_threshold is close to an edge case. This may lead to unexpected results.")
    
    dim1_original = target.shape[0]
    dim2_original = target.shape[1]
    
    assert dim1_original == dim2_original, "only quadratical images can be processed!"
    # assert (dim1_original / stride_size) % 1 == 0
    # assert (dim2_original / stride_size) % 1 == 0
    
    # before optionally padding (thereby creating more pixels from the "edge classes"), 
    # get unbiased sample weights:
    sample_weights, classes = _get_sample_weights_per_class(target, sample_weights, balanced)
    
    if padded:
        target = np.pad(target, pad_width = kernel_size - 1, mode = "edge")
        hard_labels_sampled_outputs = np.pad(hard_labels_sampled_outputs,
                                             pad_width= kernel_size - 1, 
                                             mode = "edge") #  Pads with the edge values of array.
        
    weight_matrix = _get_weight_matrix(target, classes, sample_weights)
    weights_windowed = _reshape_to_patches(weight_matrix, kernel_size, stride_size)
    
    if base_metric == "accuracy":
        arr_equal = np.equal(target, hard_labels_sampled_outputs)
        arr_windowed = _reshape_to_patches(arr_equal, kernel_size, stride_size)

        accuracy_map_pooled =  np.greater(np.average(arr_windowed,
                                                     axis=1, 
                                                     weights = weights_windowed),
                                          base_metric_threshold)
        dim1_pooled = dim2_pooled = int(np.sqrt(accuracy_map_pooled.shape[0]))
        accuracy_map_pooled = accuracy_map_pooled.reshape(dim1_pooled, dim2_pooled)
        
        return accuracy_map_pooled
    
    elif base_metric == "f1":
        target_windowed = _reshape_to_patches(target, kernel_size, stride_size)      
        hard_labels_windowed = _reshape_to_patches(hard_labels_sampled_outputs, kernel_size, stride_size)

        f1_map_pooled = _f1_per_patch(hard_labels_windowed,
                                      target_windowed, 
                                      weights=weights_windowed, 
                                    #   classes = classes, 
                                      average= average)
        
        f1_map_pooled = np.greater(f1_map_pooled, base_metric_threshold)
        
        dim1_pooled = dim2_pooled = int(np.sqrt(f1_map_pooled.shape[0]))
        # reshape back to original dims:
        f1_map_pooled = np.array(f1_map_pooled).reshape(dim1_pooled, dim2_pooled)
        return f1_map_pooled
    
    elif base_metric == "recall":
        target_windowed = _reshape_to_patches(target, kernel_size, stride_size)      
        hard_labels_windowed = _reshape_to_patches(hard_labels_sampled_outputs, kernel_size, stride_size)

        recall_map_pooled = _recall_per_patch(hard_labels_windowed,
                                              target_windowed, 
                                              weights=weights_windowed,
                                              average= average)
        
        recall_map_pooled = np.greater(recall_map_pooled, base_metric_threshold)
        
        dim1_pooled = dim2_pooled = int(np.sqrt(recall_map_pooled.shape[0]))
        # reshape back to original dims:
        recall_map_pooled = np.array(recall_map_pooled).reshape(dim1_pooled, dim2_pooled)
        return recall_map_pooled
    
    elif base_metric == "f1_sklm":
        target_windowed = _reshape_to_patches(target, kernel_size, stride_size)      
        hard_labels_windowed = _reshape_to_patches(hard_labels_sampled_outputs, kernel_size, stride_size)
        # loop over patches and see whether f1 score per patch is above threshold
        f1_map_pooled = [
            _f1_sklm(x, y, z, average = average) \
                for _, (x, y, z) in enumerate(zip(hard_labels_windowed, 
                                                  target_windowed,
                                                  weights_windowed))
                ]
        f1_map_pooled = np.greater(f1_map_pooled, base_metric_threshold)
        dim1_pooled = dim2_pooled = int(np.sqrt(len(f1_map_pooled)))
        # convert list to numpy array and reshape back to original dims:
        f1_map_pooled = np.array(f1_map_pooled).reshape(dim1_pooled, dim2_pooled)
        return f1_map_pooled
    else:
        raise ValueError("The base metric must be one of 'accuracy', 'f1', 'recall', or 'f1_sklm'.")
# -----------------------------------
def _uncertainty_map(uncertainty_type: str, 
                     prediction: np.ndarray) -> np.ndarray:
    """
    Calculate the uncertainty map of a given type for a given model prediction.

    Parameters
    ----------
    uncertainty_type : str
        The type of uncertainty to calculate. Available options: "aleatoric", "MI", or "total".
    
    prediction : np.ndarray
        The predicted output of a model. Shape: (MC_samples, num_classes, height, width).

    Returns
    -------
    np.ndarray
        The uncertainty map of the given type for the given model prediction.
        Shape: (MC_samples, height, width).
    
    Raises
    -------
    ValueError if uncertainty_type is not one of "aleatoric", "MI", or "total".
    """        
    if uncertainty_type  == "aleatoric":
        uncertainty_map = aleatoric_unc(prediction) # data uncertainty
    elif uncertainty_type == "MI":
        uncertainty_map = epistemic_unc(prediction) # mutual information: systemic/model uncertainty
    elif uncertainty_type == "total":
        uncertainty_map = total_unc(prediction) # predictive entropy
    else:
        raise ValueError('uncertainty must be either str("aleatoric") or str("MI") or str("total")')
    return uncertainty_map

# -----------------------------------   
def _uncertainty_map_pooled(uncertainty_map: np.ndarray, 
                            kernel_size: int = 2,
                            threshold: float = .0,
                            balanced: bool = False,
                            stride_size: int = None,
                            target: np.ndarray = None,
                            padded: bool = False,
                            sample_weights = None,
                            num_classes: int = None)-> np.ndarray:
    """
    Applies average-pooling to a 2D array of uncertainty values, 
    and returns a binary map of high uncertainty regions (i.e.whether a patch's mean uncertainty
    is above a given threshold).
    
    Parameters:
    -----------
        uncertainty_map: A 2D numpy array of uncertainty values.
        kernel_size: The size of the pooling kernel.
        threshold: The threshold for determining high uncertainty regions. Defaults to 0.0.
        balanced: If True, sample weights are balanced across classes.
        stride_size: The stride size for the pooling operation. Defaults to None.
        target: A 2D numpy array of ground truth labels. Defaults to None.
        padded: If True, the input is padded to maintain the same output dimensions.
        sample_weights: A 1D numpy array of sample weights. Defaults to None.
        num_classes: int denoting the number of classes. If None (default), will try to infer from target (expensive!)

    Returns
    -------
        A 2D numpy array of high uncertainty regions.
        
    Raises:
    -------
        AssertionError: If the shapes of the uncertainty map and target do not match.
        ValueError: 
            -If the threshold is outside of the possible range for uncertainty values based on entropy.
   
    Warns: 
    ----------
    -If threshold is close to or beyond an edge case.
    -If target and sample_weights are both not provided. 
    -If target is not provided and balanced is True.
    """
    if num_classes is None:
        num_classes = len(np.unique(target))
    max_unc = np.log2(num_classes)
    # Sanity checks and warnings for edge case thresholds:
    if threshold > max_unc or threshold < 0:
        # hard error since unclear how negative values etc will affect the function's calculations
        raise ValueError("threshold is outside of the  possible range for uncertainty values based on entropy.")
    if isclose(threshold, max_unc, abs_tol = 0.01) or isclose(threshold, 0.0, abs_tol = 0.01):
        # only warning since may be intentional
        warnings.warn(f"threshold is close to an edge case. This may lead to unexpected results.")
    
    if stride_size is None:
        stride_size = kernel_size
        
    if target is not None:
        assert np.array_equal(uncertainty_map.shape, target.shape), "The shapes of the uncertainty map and target must match."
        sample_weights, classes = _get_sample_weights_per_class(target, sample_weights, balanced)
    else: # weights are going to be set equally to each pixel
        if sample_weights is not None:
            warnings.warn("target not provided. sample_weights are set to equal for each pixel")
            sample_weights = None
        if balanced:
            warnings.warn("target not provided. balanced is set to False")
            balanced = False
        sample_weights, classes = _get_sample_weights_per_class(uncertainty_map, sample_weights = sample_weights, balanced = balanced)

    if padded:
        pad_width = kernel_size - 1
        uncertainty_map = np.pad(uncertainty_map, 
                                 pad_width = pad_width,
                                 mode = "edge")
        if target is not None:
            target = np.pad(target, 
                            pad_width = pad_width,
                            mode = "edge")
            
    if target is not None:
        weight_matrix = _get_weight_matrix(target, classes, sample_weights)
    else:
        weight_matrix = _get_weight_matrix(uncertainty_map, classes, sample_weights)
        
    weights_windowed = _reshape_to_patches(weight_matrix, kernel_size, stride_size)   
    uncertainty_map_windowed = _reshape_to_patches(uncertainty_map, kernel_size, stride_size)
    
    uncertainty_map_pooled =  np.greater(np.average(uncertainty_map_windowed,
                                                    axis=1, 
                                                    weights = weights_windowed),
                                         threshold)
    
    dim_pooled = int(np.sqrt(uncertainty_map_pooled.shape[0]))
    uncertainty_map_pooled = uncertainty_map_pooled.reshape(dim_pooled, dim_pooled)
    
    return uncertainty_map_pooled


# -----------------------------------  
def _f1_per_patch(y_pred: np.ndarray,
                  y_true: np.ndarray,  
                  weights: Optional[np.ndarray] = None,
                  average: str = "macro")-> np.ndarray:
    """
    Calculate the F1 score per patch for each class.
TODO
    Args:
    -------
        y_pred (np.ndarray): predicted labels of shape (n_patches, patch_height, patch_width)
        y_true (np.ndarray): ground truth labels of shape (n_patches, patch_height, patch_width)
        weights (Optional[np.ndarray]): sample weights of shape (n_patches, patch_height, patch_width) (default: None)
        average (str): type of averaging to compute the F1 score. Possible values: "macro", "weighted" (default: "macro")

    Returns:
    -------
        np.ndarray: F1 score per patch for each class of shape (n_patches, n_classes)

    Raises:
    -------
        AssertionError: if the shapes of y_pred, y_true and weights are not equal
    """
    classes = np.array(np.unique(y_true))
    n_classes_per_patch, n_occurences_class_per_patch = [], []
    for a in y_true:
        n, counts = np.unique(a, return_counts = True)
        n_classes_per_patch.append(len(n))
        # extend counts vector with zeros to be able to do: f1_per_patch_per_class * n_occurences_class_per_patch
        n_occurences_class_per_patch.append(np.pad(counts, pad_width = (0, len(classes) - len(counts)), mode = "constant") )
    n_classes_per_patch = np.array(n_classes_per_patch)
    n_occurences_class_per_patch = np.array(n_occurences_class_per_patch)
    n_patches = y_pred.shape[0]
    
    f1_per_patch_per_class = np.zeros((n_patches, len(classes)))
    
    if weights is None:
        weights = np.ones_like(y_true)
    assert np.array_equal(y_pred.shape, y_true.shape)
    assert np.array_equal(y_true.shape, weights.shape)
     
    for i in range(len(classes)):
        tp = ((np.logical_and(y_pred == i, y_true == i) * weights) / np.sum(weights)).sum(axis = 1)
        fp = ((np.logical_and(y_pred == i, y_true != i) * weights) / np.sum(weights)).sum(axis = 1)
        fn = ((np.logical_and(y_pred != i, y_true == i) * weights) / np.sum(weights)).sum(axis = 1)
        f1_per_patch_per_class[:, i] = np.nan_to_num((2 * tp) / (2 * tp + fp + fn))
    if average == "macro":
        return np.sum(f1_per_patch_per_class, axis = 1) / n_classes_per_patch
    elif average == "weighted":
        return np.sum(f1_per_patch_per_class * n_occurences_class_per_patch, axis = 1) / np.sum(n_occurences_class_per_patch, axis = 1)

# -----------------------------------
def _f1_sklm(y_pred, y_true, weights, average="macro"):
    """
    Compute the F1 score using scikit-learn's implementation.

    Args:
    -------
        y_pred (array-like): predicted labels.
        y_true (array-like): true labels.
        weights (array-like): sample weights.
        average (str, optional): Determines the type of averaging performed on the samples.
            Must be one of ('micro', 'macro', 'samples', 'weighted'). Defaults to 'macro'.

    Returns:
    -------
        float: F1 score.

    """
    return sklm.f1_score(y_pred=y_pred,
                         y_true=y_true,
                         sample_weight=weights,
                         average=average,
                         zero_division=0)

# -----------------------------------
def _recall_per_patch(y_pred: np.ndarray,
                      y_true: np.ndarray,  
                      weights: Optional[np.ndarray] = None,
                      average: str = "macro")-> np.ndarray:
    """
    TODO
    """
    classes = np.array(np.unique(y_true))
    n_classes_per_patch, n_occurences_class_per_patch = [], []
    for a in y_true:
        n, counts = np.unique(a, return_counts = True)
        n_classes_per_patch.append(len(n))
        # extend counts vector with zeros to able to do: f1_per_patch_per_class * n_occurences_class_per_patch
        n_occurences_class_per_patch.append(np.pad(counts, pad_width = (0, len(classes) - len(counts)), mode = "constant") )
    n_classes_per_patch = np.array(n_classes_per_patch)
    n_occurences_class_per_patch = np.array(n_occurences_class_per_patch)
    n_patches = y_pred.shape[0]
    
    recall_per_patch_per_class = np.zeros((n_patches, len(classes)))
    
    if weights is None:
        weights = np.ones_like(y_true)
    assert np.array_equal(y_pred.shape, y_true.shape)
    assert np.array_equal(y_true.shape, weights.shape)
     
    for i in range(len(classes)):
        tp = ((np.logical_and(y_pred == i, y_true == i) * weights) / np.sum(weights)).sum(axis = 1)
        fn = ((np.logical_and(y_pred != i, y_true == i) * weights) / np.sum(weights)).sum(axis = 1)
        recall_per_patch_per_class[:, i] = np.nan_to_num((tp) / (tp +fn))
    if average == "macro":
        return np.sum(recall_per_patch_per_class, axis = 1) / n_classes_per_patch
    elif average == "weighted":
        return np.sum(recall_per_patch_per_class * n_occurences_class_per_patch, axis = 1) / np.sum(n_occurences_class_per_patch, axis = 1)

# --------------------------
def _get_sample_weights_per_class(target: np.ndarray,
                                  sample_weights: list = None,
                                  balanced: bool = False):
    classes = np.array(np.unique(target), dtype = np.int8)
    n_classes = len(classes)
    
    if isinstance(sample_weights, list):
        sample_weights = np.array(sample_weights, dtype = np.float)
        if len(sample_weights) > n_classes:
            # this may well occur if user puts in 3 weights for 3 classes but only 1 or 2 classes are present
            # in the image. assuming that classes 0+1 are provided first and that they are the most common, anyway:
            # cut off the array and omit the weight for class 2
            warn_string = str(f'len(sample_weights) = {len(sample_weights)}, but n_classes = {n_classes}. ') + \
                str(f'Only first {n_classes} weights will be used.')
            warnings.warn(warn_string)
            sample_weights = dict.fromkeys(classes[0:n_classes])
        if len(sample_weights) < n_classes:
            # if too few class weights are provided, process is continued as if no sample_weights have been put in
            warn_string = str(f'too few weights provided: len(sample_weights) = {len(sample_weights)} < n_classes = {n_classes}.') + \
                str("replacing according to balanced = F/T option")
            warnings.warn(warn_string)
            sample_weights = None
    if sample_weights is None:
        if balanced:
            sample_weights = dict.fromkeys(classes)
            for c in iter(classes):
                sample_weights[c] = 1 / (np.equal(target, c)).sum()
            # sample_weights = [sample_weights[c] =  for c in iter(classes)]
        else:
            sample_weights = dict.fromkeys(classes)
            for c in iter(classes):
                sample_weights[c] = 1
            # sample_weights = np.array([1.0 for _ in iter(classes)])
    return sample_weights, classes

# -----------------------------------
def _get_weight_matrix(target, classes: np.ndarray, sample_weights: np.ndarray):
    """
    Generates a weight matrix for a given target array, classes and corresponding sample weights.

    Args:
    -------
        target (np.ndarray): Target array with
        classes (np.ndarray): Array with unique class labels.
        sample_weights (np.ndarray): Array with sample weights for each class.

    Returns:
    -------
        np.ndarray: A weight matrix with the same shape as the target array.
    """
    weight_matrix = np.zeros_like(target, dtype=np.float64)
    for c in classes: weight_matrix += np.equal(target, c) * sample_weights[c]
    return weight_matrix

# -----------------------------------
def _reshape_to_patches(image: np.array,
                        kernel_size: int,
                        stride_size: int) -> np.array:
    """
    Reshapes an image into patches (of size kernel_size by kernel_size) using
    a sliding window approach with a stride of stride_size. 
    The output is a 2D numpy array where each row represents a single patch.
    
    Args:
    -------
    image (numpy.ndarray): A 2D numpy array representing the input image.
    kernel_size (int): The size of the patch. A positive integer value.
    stride_size (int): The step size to move the window. A positive integer value.
    
    Returns:
    -------       
    np.array: An array of patches with shape (n_patches, kernel_size * kernel_size).

    Examples:
    -------
    [0, 1, 2, 3,
     4, 5, 6, 7] -> [[0,1,4,5], [2,3,6,7]] for kernel_size = 2, step_size = 2
    [0, 1, 2, 3,
     4, 5, 6, 7] -> [[0,1,4,5], [1,2,5,6] [2,3,6,7]] for kernel_size = 2, step_size = 1
    """
    image_windowed = view_as_windows(image,
                                     (kernel_size,kernel_size), 
                                     step = stride_size)
    n_patches = image_windowed.shape[0] * image_windowed.shape[1]

    return image_windowed.reshape(n_patches, kernel_size * kernel_size)