import numpy as np

def projectSimplex(v):
    # Computest the minimum L2-distance projection of vector v onto the probability simplex
    nVars = len(v)
    mu = sorted(v, reverse=True)#np.sort(v)[::-1]
    mu = np.array(mu)
    sm = 0
    row = 0
    sm_row = 0
    for j in range(nVars):
        sm = sm + mu[j]
        if mu[j] - (1/(j+1)) * (sm-1) > 0:
            row = j+1
            sm_row = sm
    theta = (1/row) * (sm_row-1)
    # cumulative_sum = mu.cumsum()
    # rho = np.where(mu > (cumulative_sum - 1) / (1 + np.arange(nVars)))[0][-1]
    # theta = (cumulative_sum[rho] - 1) / (rho + 1)

    # efficient as per gemini
    # cumulative_sum = mu.cumsum()  # Calculate cumulative sum
    # rho = np.where(mu > (cumulative_sum - 1) / (1 + np.arange(nVars)))[0][-1]  # Find the last index where the condition is True
    # sm_row = cumulative_sum[rho]
    # row = rho + 1  # Convert back to 1-based indexing


    # Project onto simplex
    w = np.maximum(v - theta, 0)

    return w

