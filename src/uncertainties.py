import numpy as np

def entropy(sampled_outputs: np.ndarray, 
            epsilon : float = 1e-64,
            axis : int = 1, 
            **kwargs) -> float:
    """ Calculates the entropy between (within) of a set of sampled softmax outputs.
    
    Note: by convention we set: 0 * log(0) = 0. 
            np.log2(0) returns -Inf => Add epsilon to avoid taking log of 0
     
    Parameters:
    -----------
    sampled_outputs : np.array(MC samples, one-hot encodings, dim1, dim2)
        A numpy array containing the softmax-outputs of a sampled network. The shape should be (N, C, H, W),
        where N is the number of MC samples, C is the number of classes, and H and W are the spatial dimensions.
        
    epsilon : float, optional (default=1e-64)
        A small value added to the sampled outputs to avoid taking the log of zero.

    axis : int, optional (default=1)
        The axis along which to compute the entropy. This should be the axis corresponding to the classes.

    **kwargs : dict
        Additional keyword arguments to be passed to np.sum.

    Returns:
    -------
    float
        The mean entropy (i.e., averaged over classes) of the softmax-outputs; measured in bits (due to log to basis 2).
    """
    # Add epsilon to avoid taking log of 0
    sampled_outputs = np.maximum(sampled_outputs, epsilon)
    
    # Calculate log2
    log_outputs = np.log2(sampled_outputs)

    # sampled_outputs may have contained NaN values => Replace np.log2(NaN)= -inf with 0
    log_outputs[np.isneginf(log_outputs)] = 0

    # Multiply and sum along the specified axis
    return -(sampled_outputs * log_outputs).sum(axis=axis, **kwargs)


def total_unc(sampled_outputs: np.ndarray, 
              target: np.ndarray = None, 
              average: str = "micro", 
              **kwargs)-> np.ndarray:
    """Calculate the entropy of mean Monte Carlo (MC) samples.
    
    Take (sampled) softmax outputs of a network as input and calculates the entropy of the mean predictions:
    
    1. Take average probabilities over Monte Carlo samples. 
    2. Calculate entropy of mean predictions.
    
    Note: At inference time, if "validation set" consists of 1 observation -> this is the entropy for one single input.
    
    Parameters:
    -----------
    sampled_outputs : numpy.ndarray
        Array of softmax outputs of the sampled network, with dimensions (MC samples, one-hot encodings, dim1, dim2).
        
    target : numpy.ndarray, optional
        One-hot encoding of the target variable. If specified, the function will return the predictive entropy per class.
        Default is None.
        
    average : str, optional
        Determines how the function will calculate the entropy. 
        If set to "micro" (default), the function will calculate the overall predictive entropy
        If set to "none", the function will return the predictive entropy per class. 
        
    **kwargs : 
        Additional keyword arguments to be passed to the numpy.mean function when calculating the mean output.
        
    Returns:
    -------
    numpy.ndarray
        Array of predictive entropies, with dimensions (256, 256) if target is None, or (C,) if target is specified, 
        where C is the number of classes.
    """
    if average == "none":  # return predictive entropy per class
        mean_output = sampled_outputs.mean(axis=0, keepdims = True)
        H_total = entropy(mean_output, axis = 1, keepdims = True)
        H_total_per_class = (H_total * target).mean(axis = 0)  # collapse first axis. resulting shape: (C, H, W)
        return np.mean(H_total_per_class, axis = (1, 2))  # shape (C,)
    elif average == "micro":  # default behavior
        mean_output = sampled_outputs.mean(axis=0, **kwargs)
        return entropy(mean_output, axis = 0)
    else:
        raise ValueError("average must be either 'none' or 'micro'")


def aleatoric_unc(sampled_outputs: np.ndarray,
                  target: np.ndarray = None, 
                  average: str = "micro", 
                  **kwargs) -> np.ndarray:
    """Calculate mean entropy over MC samples
    
    Parameters:
    -----------
    sampled_outputs :  np.array(MC samples, one-hot encodings, dim1, dim2)
        Softmax-outputs of sampled network
        
    target:  np.array, optional
        Target one-hot encoding labels. Required when `average` is set to `"none"` or `"weighted_by_support"`.

    average: str
        Specifies how to average the entropy metric (akin to sklearn nomenclature).
        Available options:

        - "none": Calculate the metric for each class separately, and return the metric for every class.
                  Requires that target is provided. If a class does not occur in the target, uncertainty is 0.
        - "weighted_by_support": Calculate the metric for each class separately, and weigh down by dividing through
                                 the number of class members in target. This allows to properly sum up 
                                 in a batch. If a class does not occur in the target, uncertainty is 0.
        - "micro": Calculate the metric globally, across all samples and classes.
                    Returns a matrix of the same size as the input images, where each pixel 
                    represents the uncertainty per predicted pixel. 
    Returns
    -------
    np.ndarray
        A matrix of shape (dim1, dim2) containing the mean entropy across MC samples.
    """
    if average == "weighted_by_support":
        if target is None: raise ValueError("target must be provided when average is set to 'weighted_by_support'")
        # support_0 = target[0][0].sum()
        # support_1 = target[0][1].sum()
        # support_2 = target[0][2].sum()
        # support = np.array([support_0, support_1, support_2])
        # 4 lines above can be done in 1 line:
        support = np.sum(target, axis = (0,2,3))
        
        H = entropy(sampled_outputs, axis = 1, keepdims = True)
        H_per_class = (H * target).squeeze(axis = 0) # shape: (C, H, W)
        mean_per_class = np.mean(H_per_class, axis = (1,2)) / support
        # if support == 0.0 then nan_to_num will convert the fraction to zero.
        return np.nan_to_num(mean_per_class, nan= 0.0) # shape: (C,)
    elif average == "none": # return entropy per class
        if target is None: raise ValueError("target must be provided when average is set to 'none'")
        H = entropy(sampled_outputs, axis = 1, keepdims = True)
        # shape of H: (1,1,512,512), shape of target: (1,3,512,512)
        # shape of (H * target) : (1,3,512,512)
        # Multiply H with target to get "empirical" uncertainties per (true) class.
        # Then collapse axis 0.
        # Finally, return mean aleaotric unc. per class
        H_per_class = (H * target).squeeze(axis = 0) # shape: (C, H, W)
        return np.mean(H_per_class, axis = (1,2))  # shape: (C,)
    elif average == "micro": # default behavior
        return entropy(sampled_outputs, axis = 1).mean(axis = 0, **kwargs) # shape: (J, W)
    else:
        raise ValueError("average must be either 'none', 'weighted_by_support' or 'micro'")


def epistemic_unc(sampled_outputs: np.ndarray,
                  target: np.ndarray = None, 
                  average = "micro")-> np.ndarray:
    """Calculate epistemic uncertainty
    """
    tot_unc = total_unc(sampled_outputs, target, average, keepdims= False)
    al_unc = aleatoric_unc(sampled_outputs, target, average, keepdims= False)
    assert tot_unc.shape == al_unc.shape
    return tot_unc - al_unc