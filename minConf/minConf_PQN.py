import numpy as np

from minConf.minConf_SPG import minConF_SPG
from minFunc.lbfgsUpdate import lbfgsUpdate
from minConf.lbfgsHvFunc2 import lbfgsHvFunc2
from minFunc.isLegal import isLegal
from minFunc.polyinterp import polyinterp
from minFunc.autoDif.autoGrad import autoGrad

def minConF_PQN(funObj, x, funProj, options=None):
    # function [x,f] = minConF_PQN(funObj,funProj,x,options)
    #
    # Function for using a limited-memory projected quasi-Newton to solve problems of the form
    #   min funObj(x) s.t. x in C
    #
    # The projected quasi-Newton sub-problems are solved the spectral projected
    # gradient algorithm
    #
    #   @funObj(x): function to minimize (returns gradient as second argument)
    #   @funProj(x): function that returns projection of x onto C
    #
    #   options:
    #       verbose: level of verbosity (0: no output, 1: final, 2: iter (default), 3:
    #       debug)
    #       optTol: tolerance used to check for progress (default: 1e-6)
    #       maxIter: maximum number of calls to funObj (default: 500)
    #       maxProject: maximum number of calls to funProj (default: 100000)
    #       numDiff: compute derivatives numerically (0: use user-supplied
    #       derivatives (default), 1: use finite differences, 2: use complex
    #       differentials)
    #       suffDec: sufficient decrease parameter in Armijo condition (default: 1e-4)
    #       corrections: number of lbfgs corrections to store (default: 10)
    #       adjustStep: use quadratic initialization of line search (default: 0)
    #       bbInit: initialize sub-problem with Barzilai-Borwein step (default: 1)
    #       SPGoptTol: optimality tolerance for SPG direction finding (default: 1e-6)
    #    SPGiters: maximum number of iterations for SPG direction finding (default:10)

    nVars = len(x)

    default_options = {
        'verbose': 2,           # Verbosity level
        'numDiff': 0,           # Numerical differentiation mode
        'optTol': 1e-6,         # Optimality tolerance
        'progTol': 1e-9,
        'maxIter': 500,         # Maximum outer iterations (PQN)
        'maxProject': 100000,   # Maximum projection iterations
        'suffDec': 1e-4,        # Sufficient decrease parameter for Armijo
        'corrections': 10,       # Number of L-BFGS corrections to store
        'adjustStep': 0,        # Use quadratic initialization for line search
        'bbInit': 0,            # Barzilai-Borwein step initialization for sub-problem
        'SPGoptTol': 1e-6,      # Optimality tolerance for SPG
        'SPGprogTol': 1e-10,
        'SPGiters': 10,         # Maximum inner iterations (SPG)
        'SPGtestOpt': 0         # Test optimality for SPG
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
    maxProject = options['maxProject']
    suffDec = options['suffDec']
    corrections = options['corrections']
    adjustStep = options['adjustStep']
    bbInit = options['bbInit']
    SPGoptTol = options['SPGoptTol']
    SPGprogTol = options['SPGprogTol']
    SPGiters = options['SPGiters']
    SPGtestOpt = options['SPGtestOpt']

    # Output Parameter Settings
    if verbose >= 3:
        print('Running PQN...')
        print(f'Number of L-BFGS Corrections to store: {corrections}')
        print(f'Spectral initialization of SPG: {bbInit}')
        print(f'Maximum number of SPG iterations: {SPGiters}')
        print(f'SPG optimality tolerance: {SPGoptTol:.2e}')
        print(f'SPG progress tolerance: {SPGprogTol:.2e}')
        print(f'PQN optimality tolerance: {optTol:.2e}')
        print(f'Quadratic initialization of line search: {adjustStep}')
        print(f'Maximum number of function evaluations: {maxIter}')
        print(f'Maximum number of projections: {maxProject}')

    # Output log
    if verbose >= 2:
        print(f"{'Iteration':>10} {'FunEvals':>10} {'Projections':>10} {'Step Length':>15} {'Function Val':>15} {'Opt Cond':>15}")

    # Make objective function (if using numerical derivatives)
    funEvalMultiplier = 1
    if numDiff:
        if numDiff == 2:
            useComplex = 1
        else:
            useComplex = 0
        funObj = lambda x: autoGrad(x, useComplex, funObj)
        funEvalMultiplier = nVars + 1 - useComplex

    # Project initial parameter vector
    x = funProj(x)
    projects = 1

    # Evaluate initial parameters
    f, g = funObj(x)
    funEvals = 1

    # Check Optimality of Initial Point
    projects = projects + 1
    if np.max(np.abs(funProj(x - g) - x)) < optTol:
        if verbose >= 1:
            print('First-Order Optimality Conditions Below optTol at Initial Point')
        return None
    
    i = 1
    while funEvals <= maxIter:
        # Compute Step Direction
        if i == 1:
            p = funProj(x-g)
            projects = projects + 1
            S = np.zeros((nVars, 0))
            Y = np.zeros((nVars, 0))
            Hdiag = 1
        else:
            y = g - g_old
            s = x - x_old
            S, Y, Hdiag = lbfgsUpdate(y, s, corrections, verbose==3, S, Y, Hdiag)

            # Make Compact Representation
            k = Y.shape[1]
            L = np.zeros((k, k))
            for j in range(k):
                # L[j+1:k, j+1] = S[:, j+1:k+1].T @ Y[:, j+1]
                L[j+1:k, j] = S[:, j+1:k].T @ Y[:, j]
            N = np.hstack((S / Hdiag, Y))  # Horizontal stacking of arrays
            M = np.block([
                [S.T @ S / Hdiag, L],
                [L.T, -np.diag(np.diag(S.T @ Y))]
            ])
            HvFunc = lambda v: lbfgsHvFunc2(v, Hdiag, N, M)

            if bbInit:
                # Use Barzilai-Borwein step to initialize sub-problem
                alpha = (S.T @ S)/(S.T @ y)
                if alpha <= 1e-10 or alpha > 1e10:
                    alpha = np.minimum(1, 1/np.sum(np.abs(g)))
                    # alpha = 1/np.linalg.norm(g)

                # Solve Sub-problem
                xSubInit = x-alpha @ g
                feasibleInit = 0
            else:
                xSubInit = x
                feasibleInit = 1

            # Solve Sub-problem
            p, subProjects = solveSubProblem(x, g, HvFunc, funProj, SPGoptTol, SPGprogTol, SPGiters, SPGtestOpt, feasibleInit, xSubInit)
            projects = projects + subProjects

        d = p - x
        g_old = g
        x_old = x

        # Check that Progress can be made along the direction
        gtd = g.T @ d
        if gtd > -optTol:#progTol:
            if verbose >= 1:
                print('Directional Derivative below optTol')
            break

        # Select Initial Guess to step length
        if i==1 or adjustStep==0:
            t = 1
        else:
            t = min(1, 2*(f-f_old)/gtd)

        # Bound Step length on first iteration
        if i == 1:
            t = min(1, 1 / np.sum(np.abs(g)))

        # Evaluate the Objective and Gradient at the Initial Step
        if t==1:
            x_new = p
        else:
            x_new = x + t * d

        f_new, g_new = funObj(x_new)
        funEvals = funEvals + 1

        # Backtracking Line Search
        f_old = f
        while f_new.item() > (f + suffDec * g.T @ (x_new - x)).item() or not isLegal(f_new):
            temp = t

            # Backtrack to next trial value
            if not isLegal(f_new) or not isLegal(g_new):
                if verbose == 3:
                    print('Halving Step Size')
                t = t/2
            else:
                if verbose == 3:
                    print('Cubic Backtracking')
                t = polyinterp(points=np.array([[0, f.item(), gtd.item()], [t, f_new.item(), (g_new.T @ d).item()]]))

            # Adjust if change is too small/large
            if t < temp * 1e-3:
                if verbose == 3:
                    print('Interpolated value too small, Adjusting')
                t = temp * 1e-3
            elif t > temp * 0.6:
                if verbose == 3:
                    print('Interpolated value too large, Adjusting')
                t = temp * 0.6

            # Check whether step has become too small
            if np.sum(np.abs(t * d)) < progTol or t == 0:
                if verbose == 3:
                    print('Line Search failed')
                t = 0
                f_new = f
                g_new = g
                break

            # Evaluate new point
            f_prev = f_new
            t_prev = temp
            x_new = x + t * d # change from x_new = x + t @ d
            f_new, g_new = funObj(x_new)
            funEvals = funEvals + 1

        # Take step
        x = x_new
        f = f_new
        g = g_new

        
        optCond = np.sum(np.abs(funProj(x - g) - x))#np.max(np.abs(funProj(x - g) - x))
        projects = projects + 1

        # Output log
        if verbose >= 2:
            print(f"{i:>10d} {funEvals:>10d} {projects:>10d} {t:>15.5e} {f.item():>15.5e} {optCond:>15.5e}")

        # Check optimality
        if optCond < optTol:
            print('First-Order Optimality Conditions Below optTol')
            break

        if np.max(np.abs(t * d)) < progTol:
            if verbose >= 1:
                print('Step size below progTol')
            break

        if np.abs(f - f_old) < progTol:
            if verbose >= 1:
                print('Function value changing by less than progTol')
            break

        if funEvals*funEvalMultiplier > maxIter:
            if verbose >= 1:
                print('Function Evaluations exceeds maxIter')
            break

        if projects > maxProject:
            if verbose >= 1:
                print('Number of projections exceeds maxProject')
            break

        i += 1
    

    return x, f, funEvals

def solveSubProblem(x, g, H, funProj, optTol, progTol, maxIter, testOpt, feasibleInit, x_init):
    # Uses SPG to solve for projected quasi-Newton direction
    options = {
        'verbose': 0,
        'optTol': optTol,
        'progTol': progTol,
        'maxIter': maxIter,
        'testOpt': testOpt,
        'feasibleInit': feasibleInit
    }

    funObj = lambda p: subHv(p, x, g, H)
    p, f, funEvals, subProjects = minConF_SPG(funObj, x_init, funProj, options)
    return p, subProjects

def subHv(p, x, g, HvFunc):
    d = p - x
    Hd = HvFunc(d)
    f = g.T @ d + 0.5 * d.T @ Hd
    g = g + Hd
    return f, g