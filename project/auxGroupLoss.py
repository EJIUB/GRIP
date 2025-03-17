import numpy as np

def auxGroupLoss(w, groups, lamda, funObj, nargout=2):
    """
    Calculates the auxiliary group loss, gradient, and Hessian (if requested).

    Args:
        w (np.ndarray): Weight vector.
        groups (np.ndarray): Group assignments for features.
        lambda_ (float or np.ndarray): Regularization parameter(s).
        funObj (callable): Objective function (returns value, gradient, and optionally Hessian).
        nargout (int, optional): Number of output arguments requested (default: 2).

    Returns:
        f (float): Loss value.
        g (np.ndarray): Gradient vector.
        H (np.ndarray, optional): Hessian matrix (if nargout == 3).
    """
    p = len(groups)
    nGroups = len(w) - p
    
    if nargout == 3:
        f, g, H = funObj(w[:p])  
    else:
        f, g = funObj(w[:p])

    f = f + np.sum(lamda * w[p:])
    g = np.concatenate([g, lamda * np.ones(nGroups)])
 
    if nargout == 3:
        H_zero1 = np.zeros((p, nGroups))
        H_zero2 = np.zeros((nGroups, p + nGroups))
        H = np.block([[H, H_zero1], [H_zero2]])

    if nargout == 2:  
        return f, g
    return f, g, H
