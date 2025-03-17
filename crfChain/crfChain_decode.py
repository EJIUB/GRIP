import numpy as np

def crfChain_decode(nodePot, edgePot):
    nNodes, nStates = nodePot.shape

    # Forward Pass
    alpha = np.zeros((nNodes, nStates))
    Z = np.zeros(nNodes)
    mxState = np.zeros((nNodes, nStates), dtype=int)

    alpha[0, :] = nodePot[0, :]
    Z[0] = np.sum(alpha[0, :])
    alpha[0, :] /= Z[0]

    for n in range(1, nNodes):
        tmp = np.tile(alpha[n-1, :].reshape(-1, 1), (1, nStates)) * edgePot
        alpha[n, :] = nodePot[n, :] * np.max(tmp, axis=0)
        mxState[n, :] = np.argmax(tmp, axis=0)
        
        # Normalize
        Z[n] = np.sum(alpha[n, :])
        alpha[n, :] /= Z[n]

    # Backward Pass
    y = np.zeros(nNodes, dtype=int)
    y[-1] = np.argmax(alpha[-1, :])
    
    for n in range(nNodes-2, -1, -1):
        y[n] = mxState[n+1, y[n+1]]

    return y


