import numpy as np

def lbfgsUpdate(y, s, corrections, debug, old_dirs, old_stps, Hdiag):
    ys = y.conj().T @ s
    if ys > 1e-10:
        num_corrections = old_dirs.shape[1]
        if num_corrections < corrections:
            # Full Update
            old_dirs = np.column_stack((old_dirs, s))
            old_stps = np.column_stack((old_stps, y))
        else:
            # Limited-Memory Update
            old_dirs = np.column_stack((old_dirs[:, 1:corrections], s))
            old_stps = np.column_stack((old_stps[:, 1:corrections], y))

        # Update scale of initial Hessian approximation
        Hdiag = ys / (y.conj().T @ y)
    else:
        if debug:
            print('Skipping Update')

    return old_dirs, old_stps, Hdiag