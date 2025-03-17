import numpy as np

def groupl1_makeGroupPointers(groups):
    """
    Creates group start indices and element pointers for grouped L1 regularization.

    Args:
        groups (np.ndarray): A 1D array of integers representing group assignments for elements.

    Returns:
        groupStart (np.ndarray): Array of starting indices for each group in the data.
        groupPtr (np.ndarray): Array of indices pointing to elements within each group.
        nGroups (int): The total number of groups.
    """

    nVars = len(groups)
    nGroups = np.max(groups)
    print(nGroups, nVars)
    input()

    # Count elements in each group
    groupStart = np.zeros(nGroups + 1, dtype=int)  # Initialize with zeros
    for i in range(nVars):
        if groups[i] > 0:  # Python is 0-indexed, no need to add 1 here
            groupStart[groups[i]] += 1
    groupStart[0] = 0  # First group starts at index 0
    groupStart = np.cumsum(groupStart)  # Cumulative sum

    # Create pointers to group elements
    groupPtr = np.zeros((nVars, 1), dtype=int)
    groupPtrInd = np.zeros((nGroups, 1), dtype=int)
    print(groupPtrInd.shape)
    for i in range(nVars):
        if groups[i] > 0:
            grp = groups[i]
            groupPtr[groupStart[grp] + groupPtrInd[grp]] = i
            groupPtrInd[grp] += 1

    return groupStart, groupPtr, nGroups
