import numpy as np

from minFunc.autoDif.autoGrad import autoGrad
from minFunc.isLegal import isLegal
from minFunc.polyinterp import polyinterp

def minConF_SPG(funObj, x, funProj, options):
    nVars = len(x)

    default_options = {
        'verbose': 2,           # Verbosity level
        'numDiff': 0,           # Numerical differentiation mode
        'optTol': 1e-5,         # Optimality tolerance
        'progTol': 1e-9,
        'maxIter': 500,         # Maximum outer iterations (PQN)
        'suffDec': 1e-4,        # Sufficient decrease parameter for Armijo
        'interp': 2,       # Number of L-BFGS corrections to store
        'memory': 10,        # Use quadratic initialization for line search
        'useSpectral': 1,            # Barzilai-Borwein step initialization for sub-problem
        'curvilinear': 0,      # Optimality tolerance for SPG
        'feasibleInit': 0,         # Maximum inner iterations (SPG)
        'testOpt': 1,         # Test optimality for SPG
        'bbType': 1
    }

    # Merge user-provided options with defaults
    if options is None:
        options = {}
    options = {**default_options, **options}

    verbose = options['verbose']
    numDiff = options['numDiff']
    optTol = options['optTol']
    progTol = options['progTol']
    maxIter = options['maxIter']
    suffDec = options['suffDec']
    interp = options['interp']
    memory = options['memory']
    useSpectral = options['useSpectral']
    curvilinear = options['curvilinear']
    feasibleInit = options['feasibleInit']
    testOpt = options['testOpt']
    bbType = options['bbType']

    # Output Log
    if verbose >= 2:
        if testOpt:
            print(f"{'Iteration':>10} {'FunEvals':>10} {'Projections':>10} {'Step Length':>15} {'Function Val':>15} {'Opt Cond':>15}")
        else:
            print(f"{'Iteration':>10} {'FunEvals':>10} {'Projections':>10} {'Step Length':>15} {'Function Val':>15}")

    # Make objective function (if using numerical derivatives)
    funEvalMultiplier = 1
    if numDiff:
        if numDiff==2:
            useComplex = 1
        else:
            useComplex = 0
        funObj = lambda x: autoGrad(x, useComplex, funObj)
        funEvalMultiplier = nVars + 1 - useComplex

    # Evaluate Initial Point
    if not feasibleInit:
        x = funProj(x)
    
    f, g = funObj(x)
    projects = 1
    funEvals = 1

    # Optionally check optimality
    if testOpt:
        projects = projects + 1
        if np.max(np.abs(funProj(x-g)-x)) < optTol:
        # if np.sum(np.abs(funProj(x-g)-x)) < optTol:
            if verbose >= 1:
                print("First-Order Optimality Conditions Below optTol at Initial Point")
            return None#x, f, funEvals
        
    i=1
    while funEvals <= maxIter:
        # Compute Step Direction
        if i==1 or not useSpectral:
            alpha = 1
        else:
            y = g - g_old
            s = x - x_old
            if bbType==1:
                alpha = (s.conj().T @ s) / (s.conj().T @ y)
            else:
                alpha = (s.conj().T @ y) / (y.conj().T @ y)
            
            if alpha <= 1e-10 or alpha > 1e10:
                alpha = 1
        
        d = -alpha * g
        f_old = f
        x_old = x
        g_old = g

        # Compute Projected Step
        if not curvilinear:
            d = funProj(x+d) - x
            projects = projects + 1

        # Check that Progress can be made along the direction
        gtd = g.conj().T @ d
        if gtd > -progTol:
            if verbose >= 1:
                print("Directional Derivative below progTol")
            break

        # Select Initial Guess to step length
        if i==1:
            t = min(1, 1 / np.sum(np.abs(g)))
        else:
            t = 1
        
        # Compute reference function for non-monotone condition

        if memory == 1:
            funRef = f
        else:
            if i == 1:
                old_fvals = np.full((memory, 1), -np.inf)  
            
            if i <= memory:
                old_fvals[i-1] = f
            else:
                old_fvals = np.vstack((old_fvals[1:], f)) # Remove the first element and append the new one
            funRef = np.max(old_fvals)

        
        # Evaluate the objective and Gradient at the Initial Step
        if curvilinear:
            x_new = funProj(x + t*d)
            projects = projects + 1
        else:
            x_new = x + t*d
        f_new, g_new = funObj(x_new)
        funEvals = funEvals + 1

        # Backtracking Line Search
        lineSearchIters = 1
        while f_new > funRef + suffDec*g.conj().T@(x_new - x) or not isLegal(f_new):
            temp = t
            if interp == 0 or not isLegal(f_new):
                if verbose == 3:
                    print("Halving Step Size")
                t = t/2
            elif interp == 2 and isLegal(g_new):
                if verbose == 3:
                    print("Cubic Backtracking")
                t = polyinterp(points=np.array([[0, f.item(), gtd.item()], [t, f_new.item(), (g_new.conj().T@d).item()]]))
            elif lineSearchIters < 2 or not isLegal(f_prev):
                if verbose == 3:
                    print("QUadratic Backtracking")
                
                t = polyinterp(points=np.array([[0, f, gtd], [t, f_new, np.sqrt(-1)]]))
            else:
                if verbose == 3:
                    print("Cubic Backtracking on Function Values")
                t = polyinterp(points=np.array([[0, f, gtd], [t, f_new, np.sqrt(-1)], [t_prev, f_prev, np.sqrt(-1)]]))
            
            # Adjust if change is too small
            if t<temp*1e-3:
                if verbose==3:
                    print("Interpolated value too small, Adjusting")
                t = temp*1e-3
            elif t>temp*0.6:
                if verbose==3:
                    print("Interpolated value too large, Adjusting")
                t = temp*0.6
            
            # Check whether step has become too small
            if np.max(np.abs(t*d)) < progTol or t==0:#np.sum(np.abs(t*d)) < optTol or t==0:
                if verbose==3:
                    print("Line Search Failed")
                t = 0
                f_new = f
                g_new = g
                break

            # Evaluate New Point
            f_prev = f_new
            t_prev = temp
            if curvilinear:
                x_new = funProj(x + t*d)
                projects = projects + 1
            else:
                x_new = x + t*d
            f_new, g_new = funObj(x_new)
            funEvals = funEvals + 1
            lineSearchIters = lineSearchIters + 1

        # Take Step
        x = x_new
        f = f_new
        g = g_new

        if testOpt:
            optCond = np.max(np.abs(funProj(x-g)-x))#np.sum(np.abs(funProj(x-g)-x))
            projects = projects + 1
        
        # Output Log
        if verbose >= 2:
            if testOpt:
                print(f'{i:10d} {funEvals * funEvalMultiplier:10d} {projects:10d} {t:15.5e} {f:15.5e} {optCond:15.5e}')
            else:
                print(f'{i:10d} {funEvals * funEvalMultiplier:10d} {projects:10d} {t:15.5e} {f:15.5e}')

        # Check optimality
        if testOpt:
            if optCond < optTol:
                if verbose >= 1:
                    print('First-Order Optimality Conditions Below optTol')
                break

        if np.max(np.abs(t*d)) < progTol:#np.sum(np.abs(t * d)) < optTol:
            if verbose >= 1:
                print('Step size below progTol')
            break

        if np.abs(f - f_old) < progTol:
            if verbose >= 1:
                print('Function value changing by less than progTol')
            break

        if funEvals * funEvalMultiplier > maxIter:
            if verbose >= 1:
                print('Function Evaluations exceeds maxIter')
            break

        i += 1
    return x, f, funEvals, projects

