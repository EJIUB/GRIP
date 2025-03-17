import numpy as np

def SimultaneousSquaredError(W, X, Y, nargout=1):
    """
    Calculates the simultaneous squared error and gradient for linear regression.

    Args:
        W (np.ndarray): Weight matrix.
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Target matrix.

    Returns:
        f (float): Simultaneous squared error.
        g (np.ndarray, optional): Gradient vector, returned if requested.
    """

    W = W.reshape((X.shape[1], Y.shape[1]))  # Ensure W has correct shape

    XW = X @ W       # Matrix multiplication using '@'
    res = XW - Y
    f = np.sum(res**2)  # Squared error

    if nargout > 1:  # Check if gradient is requested
        g = 2 * (X.T @ res)  # Calculate gradient
        g = g.flatten()   # Flatten into a 1D array
        return f, g
    return f
