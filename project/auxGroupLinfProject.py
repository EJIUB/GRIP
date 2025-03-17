import numpy as np

def auxGroupLinfProject(w, p, groupStart, groupPtr):
    """
    Projects a vector onto the group L-infinity norm ball.

    Args:
        w: The vector to project.
        p: The number of initial elements in w not subject to group constraints.
        groupStart: Array of starting indices for each group.
        groupPtr: Array of indices pointing to elements within each group.

    Returns:
        The projected vector.
    """
    alpha = w[p:]  
    w = w[:p]

    for i in range(len(groupStart) - 1):
        groupInd = groupPtr[groupStart[i]:groupStart[i + 1]] - 1  # 0-based indexing adjustment
        w[groupInd], alpha[i] = projectAuxSort(w[groupInd], alpha[i])
    w = np.concatenate([w, alpha])
    return w


def projectAuxSort(w, alpha):
    """
    Helper function to project a single group.
    """
    if not np.all(np.abs(w) <= alpha):
        sorted_w = np.sort(np.abs(w))[::-1]  # Sort in descending order
        sorted_w = np.append(sorted_w, 0)
        s = 0
        for k in range(len(sorted_w)):
            s += sorted_w[k]
            projPoint = (s + alpha) / (k + 1)

            if projPoint > 0 and projPoint > sorted_w[k + 1]:
                w[np.abs(w) >= sorted_w[k]] = np.sign(w[np.abs(w) >= sorted_w[k]]) * projPoint
                alpha = projPoint
                break

            if k == len(sorted_w) - 1:
                w = np.zeros_like(w)
                alpha = 0

    return w, alpha
