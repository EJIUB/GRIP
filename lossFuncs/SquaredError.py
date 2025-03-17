import numpy as np

def SquaredError(w, X, y, nargout):
    # w(feature, 1)
    # X(instance, feature)
    # y(instance, 1)

    if nargout < 3:
        # Use 2 matrix-vector products with X
        Xw = X @ w
        res = Xw - y
        f = np.sum(res**2)

        if nargout > 1:
            g = 2*(X.T @ res)
            return f, g
        return f
    else:
        # Explicitly form X'X and do 2 matrix-vector product
        n, p = X.shape
        XX = X.T @ X # np^2
        
        if n < p: # Do two matrix-vector products with X
            Xw = X @ w
            res = Xw - y
            f = np.sum(res**2)
            g = 2*(X.T @ res)
        else:
            XXw = XX @ w
            Xy = X.T @ y
            f = w.T @ XXw - 2 * w.T @ Xy + y.T @ y
            g = 2 * XXw - 2 * Xy

        H = 2 * XX
        if nargout > 3:
            p = len(w)
            T = np.zeros((p, p, p))
            return f, g, H, T

        return f, g, H
