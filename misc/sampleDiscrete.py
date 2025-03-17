import numpy as np
from numpy.random import rand

def sampleDiscrete(p):
    # Returns a sample from a discrete probability mass function indexed by p
    U = rand()
    u = 0
    for i in range(len(p)):
        u += p[i]
        if u > U:
            return i
    return len(p)