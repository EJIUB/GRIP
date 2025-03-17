def autoGrad(x, typeDiff, funObj, *args):
    # [f,g] = autoGrad(x,useComplex,funObj,varargin)
    #
    # Numerically compute gradient of objective function from function values
    #
    # type =
    #     1 - forward-differencing (p+1 evaluations)
    #     2 - central-differencing (more accurate, but requires 2p evaluations)
    #     3 - complex-step derivative (most accurate and only requires p evaluations, but only works for certain objectives)

    p = len(x)

    if typeDiff == 1: # Use Finite Differencing
        f = funObj(x, args)