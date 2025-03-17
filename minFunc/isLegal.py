import numpy as np

def isLegal(v):
    return not (np.any(np.iscomplex(v)) or np.any(np.isnan(v)) or np.any(np.isinf(v)))