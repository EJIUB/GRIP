import numpy as np

def lbfgsHvFunc2(v, Hdiag, N, M):
    Hv = v / Hdiag

    N_transpose_v = N.conj().T @ v

    solution = np.linalg.solve(M, N_transpose_v)

    N_solution = N @ solution

    Hv -= N_solution
    return Hv