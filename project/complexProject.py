import numpy as np

from mexAll import projectRandom2C

def complexProject(xy, tau):
    z = np.vectorize(complex)(xy[:len(xy)//2], xy[len(xy)//2:])
    p_z = np.sign(z) * projectRandom2C(np.abs(z), tau)
    p_xy = np.concatenate((np.real(p_z), np.imag(p_z)))
    return p_xy
