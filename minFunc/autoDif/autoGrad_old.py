import numpy as np

def autoGrad(x, useComplex, funObj, *args):
    """
    Numerically compute gradient of objective function from function values.

    Parameters:
    x - numpy array, point at which to evaluate the gradient
    useComplex - boolean, whether to use complex differentials
    funObj - function, objective function
    args - additional arguments to pass to funObj

    Returns:
    f - function value at x
    g - gradient at x
    """
    p = len(x)
    mu = 1e-150

    if useComplex:  # Use Complex Differentials
        diff = np.zeros(p, dtype=complex)
        for j in range(p):
            e_j = np.zeros(p)
            e_j[j] = 1
            diff[j] = funObj(x + mu * 1j * e_j, *args)

        f = np.mean(np.real(diff))
        g = np.imag(diff) / mu
    else:  # Use Finite Differencing
        f = funObj(x, *args)
        mu = 2 * np.sqrt(1e-12) * (1 + np.linalg.norm(x)) / np.linalg.norm(p)
        diff = np.zeros(p)
        for j in range(p):
            e_j = np.zeros(p)
            e_j[j] = 1
            diff[j] = funObj(x + mu * e_j, *args)
        g = (diff - f) / mu

    return f, g