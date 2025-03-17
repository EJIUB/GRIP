import numpy as np

def groupLinfProj(x, tau, groups):
    nGroups = int(np.max(groups))  # Python is 0-indexed
    # print(f"nGroups: {nGroups}\n groups: {groups}\n tau: {tau}")
    # input()
    # groups = groups.reshape(-1, 1)

    # Compute normalized distances across all groups
    groupVars, mx, dist = {}, np.zeros(nGroups), {}
    for g in range(nGroups):
        # Select elements of x based on groups == g, take absolute values, and append 0
        groupVars[g] = sorted(np.concatenate((np.abs(x[groups == g+1]), [0])), reverse=True)
        
        # Calculate mx
        mx[g] = groupVars[g][0]
        
        # Initialize dist[g]
        dist[g] = np.zeros((len(groupVars[g]), 1))
        dist[g][0, 0] = 0
        
        # Calculate remaining elements of dist[g]
        for v in range(2, len(groupVars[g]) + 1):
            dist[g][v - 1, 0] = dist[g][v - 2, 0] + (v - 1) * (groupVars[g][v - 2] - groupVars[g][v - 1])
    
    # print(f"groupVars: {groupVars}\n mx: {mx}\n dist: {dist}")
    # input()

    # Check trivial case
    if np.sum(mx) <= tau:
        return x

    # Sort normalized distances
    allDist = np.zeros((0, 1))

    # Accumulate distances from each group into allDist
    for g in range(nGroups):
        allDist = np.vstack((allDist, dist[g]))#allDist = np.concatenate((allDist, dist[g]))
    # allDist = np.concatenate([dist[g] for g in range(nGroups)])
    allDist = sorted(allDist, reverse=True)#np.sort(allDist)[::-1]  
    minD, maxD = 1, len(allDist)

    # Binary search for distance that brackets tau
    # while True:
    #     ind = ind = np.floor((maxD + minD) / 2).astype(int)#(maxD + minD) // 2
    #     D = allDist[ind]  # Python indexing starts at 0

    #     for g in range(nGroups):
    #         tmp = np.max(np.where(dist[g] <= D))  
    #         if tmp == len(groupVars[g]) - 1:  # Check if tmp is the last index
    #             mx[g] = 0
    #         else:
    #             relativePos = (D - dist[g][tmp]) / (dist[g][tmp+1] - dist[g][tmp])
    #             mx[g] = (groupVars[g][tmp+1] - groupVars[g][tmp]) * relativePos + groupVars[g][tmp]

    #     L1Linf = np.sum(mx)

    #     if tau < L1Linf:
    #         maxD = ind - 1
    #     else:
    #         if ind == len(allDist):
    #             D2 = 0
    #         else:
    #             D2 = allDist[ind]  

    #         mx2 = np.zeros(nGroups)
    #         for g in range(nGroups):
    #             tmp = np.max(np.where(dist[g] <= D2)[0])
    #             if (tmp.size == 0) or tmp == len(groupVars[g]):
    #                 mx2[g] = 0
    #             else:
    #                 relativePos = (D2 - dist[g][tmp-1]) / (dist[g][tmp] - dist[g][tmp-1])
    #                 mx2[g] = (groupVars[g][tmp] - groupVars[g][tmp-1]) * relativePos + groupVars[g][tmp-1]
    #         L1Linf2 = np.sum(mx2)

    #         if tau > L1Linf2:
    #             minD = ind + 1
    #         else:
    #             break

    while True:
        ind = (maxD + minD) // 2
        D = allDist[ind]

        for g in range(nGroups):
            tmp = np.where(dist[g] <= D)[0]
            if len(tmp) == 0 or tmp[-1] == len(groupVars[g]) - 1:
                mx[g] = 0
            else:
                tmp = tmp[-1]  # Get the last index
                relativePos = (D - dist[g][tmp]) / (dist[g][tmp + 1] - dist[g][tmp])
                mx[g] = (groupVars[g][tmp + 1] - groupVars[g][tmp]) * relativePos + groupVars[g][tmp]

        L1Linf = np.sum(mx)

        if tau < L1Linf:
            maxD = ind - 1
        else:
            if ind == len(allDist) - 1:
                D2 = 0
            else:
                D2 = allDist[ind + 1]
            mx2 = np.zeros(nGroups)
            for g in range(nGroups):
                tmp = np.where(dist[g] <= D2)[0]
                if len(tmp) == 0 or tmp[-1] == len(groupVars[g]) - 1:
                    mx2[g] = 0
                else:
                    tmp = tmp[-1]  # Get the last index
                    relativePos = (D2 - dist[g][tmp]) / (dist[g][tmp + 1] - dist[g][tmp])
                    mx2[g] = (groupVars[g][tmp + 1] - groupVars[g][tmp]) * relativePos + groupVars[g][tmp]

            L1Linf2 = np.sum(mx2)

            if tau > L1Linf2:
                minD = ind + 1
            else:
                break

    # Form final result
    p = x#.copy() 
    if L1Linf2 != L1Linf:
        mu = (tau - L1Linf) / (L1Linf2 - L1Linf)
    else:
        mu = 0

    # print(f"p: {p}\n mx: {mx}\n mx2: {mx2}")
    # input()
    
    # for g in range(nGroups):
    #     groupMax = mx[g] + mu * (mx2[g] - mx[g])
    #     groupVars = x[groups == g+1]
    #     print(groupVars)
    #     input()
    #     violating = np.abs(groupVars) > groupMax
    #     groupVars[violating] = np.sign(groupVars[violating]) * groupMax
    #     p[groups == g] = groupVars
    for g in range(nGroups):
        # print(f"mx[{g}]: {mx[g]}\n mu: {mu}\n (mx2[{g}]-mx[{g}]: {mx2[g]-mx[g]})")
        # input()
        groupMax = mx[g] + mu * (mx2[g] - mx[g])
        # print(f"{groupMax}")
        groupVars = x[groups == g+1]
        # print(f"{groupVars.shape}\n {groupMax.shape}")
        # input()
        violating = np.abs(groupVars) > groupMax
        groupVars[violating] = np.sign(groupVars[violating]) * groupMax
        
        p[groups == g+1] = groupVars

    return p
