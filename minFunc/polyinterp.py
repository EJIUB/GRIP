import numpy as np
import matplotlib.pyplot as plt

def polyinterp(points, doPlot=None, xminBound=None, xmaxBound=None):
    """
    Finds the minimum of an interpolating polynomial based on function and derivative values.

    Args:
        points: A NumPy array of shape (pointNum, 3) where each row represents [x, f, g].
        doPlot: If True, plots the points, derivatives, and the interpolating polynomial.
        xminBound: Minimum value that brackets the minimum (default: minimum of points).
        xmaxBound: Maximum value that brackets the maximum (default: maximum of points).

    Returns:
        minPos: The x-coordinate of the minimum.
        fmin: The minimum value of the interpolating polynomial.
    """

    if doPlot==None and xminBound==None and xmaxBound==None:
        doPlot = True

    nPoints = points.shape[0]
    order = np.sum(np.isreal(points[:, 1:3])) - 1  # Count known f and g values
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])

    if xminBound is None:
        xminBound = xmin
    if xmaxBound is None:
        xmaxBound = xmax

    # Special case: cubic interpolation of 2 points with function and derivative values
    if nPoints == 2 and order == 3 and not doPlot==False:
        minVal, minPos = np.min(points[:, 1]), np.argmin(points[:, 1])
        # notMinPos = -minPos+3
        notMinPos = 1 - minPos
        d1 = points[minPos, 2] + points[notMinPos, 2] - 3 * (points[minPos, 1] - points[notMinPos, 1]) / (points[minPos, 0] - points[notMinPos, 0])
        d2 = np.sqrt(d1**2 - points[minPos, 2] * points[notMinPos, 2])
        if np.isreal(d2):
            t = points[notMinPos, 0] - (points[notMinPos, 0] - points[minPos, 0]) * ((points[notMinPos, 2] + d2 - d1) / (points[notMinPos, 2] - points[minPos, 2] + 2 * d2))
            minPos = np.clip(t, xminBound, xmaxBound)  # Ensure minPos is within bounds
        else:
            minPos = (xmaxBound + xminBound) / 2
        return minPos#, np.polyval(np.polyfit(points[:, 0], points[:, 1], order), minPos)

    # Constraints based on available function values
    A = np.zeros((0, order + 1))
    b = np.zeros((0, 1))
    for i in range(nPoints):
        if np.isreal(points[i, 2]):
            constraint = np.zeros((1, order + 1))
            print(constraint, constraint.shape, order)
            input("pause...")
            for j in range(order, -1, -1):
                if j == 0:
                    constraint[0, j] = 1   # Force 0^0 to be 1
                else:
                    constraint[0, j] = points[i, 0] ** j
                # constraint[0, order-j+1] = points[i, 0]**j
                print(constraint)
                input("pause...")
            # constraint = np.array([points[i, 0]**j for j in range(order, -1, -1)])
            A = np.vstack([A, constraint])
            b = np.vstack([b, points[i, 1]])

    # Constraints based on available derivatives
    for i in range(nPoints):
        if np.isreal(points[i, 2]):
            constraint = np.array([(order - j + 1) * points[i, 0]**(order - j) for j in range(1, order + 1)])
            A = np.vstack([A, constraint])
            b = np.vstack([b, points[i, 2]])

    # Find interpolating polynomial
    params = np.linalg.lstsq(A, b, rcond=None)[0] 

    # Compute critical points 
    dParams = np.array([params[i] * (order - i) for i in range(len(params) - 1)])
    if np.any(np.isinf(dParams)) or np.any(np.isnan(dParams)):
        cp = np.array([xminBound, xmaxBound, *points[:, 0]])
    else:
        cp = np.array([xminBound, xmaxBound, *points[:, 0], *np.roots(dParams)])

    # Test critical points and find the minimum
    fmin = np.inf
    minPos = (xminBound + xmaxBound) / 2
    for xCP in cp:
        if np.isreal(xCP) and xminBound <= xCP <= xmaxBound:
            fCP = np.polyval(params, xCP)
            if np.isreal(fCP) and fCP < fmin:
                minPos = np.real(xCP)
                fmin = np.real(fCP)
 
    # Plotting (if requested)
    if doPlot:
        plt.figure()  # Create a new figure
        plt.plot(points[:, 0], points[:, 1], 'b*') # Plot points
        for i in range(nPoints):
            if np.isreal(points[i, 2]):
                m, b = points[i, 2], points[i, 1] - m * points[i, 0]
                x_vals = [points[i, 0] - 0.05, points[i, 0] + 0.05]
                y_vals = [m * x + b for x in x_vals]
                plt.plot(x_vals, y_vals, 'c.-')  # Plot derivative lines
        # Plot interpolated polynomial
        x = np.linspace(min(xmin, xminBound) - 0.1, max(xmax, xmaxBound) + 0.1, 100)
        f = np.polyval(params, x)
        plt.plot(x, f, 'y')
        plt.plot(minPos, fmin, 'g+')  # Mark the minimum
        plt.axis([x[0] - 0.1, x[-1] + 0.1, min(f) - 0.1, max(f) + 0.1])
        plt.show()

    return minPos, fmin

# Example usage
# points = np.array([[-1, 2, -1], [1, 0, 1]])
# min_pos, f_min = polyinterp(points, doPlot=True)
# print("Minimum Position:", min_pos)
# print("Minimum Value:", f_min)



# import numpy as np
# import matplotlib.pyplot as plt

# def polyinterp(**kwargs):
#     if len(kwargs) < 2:
#         kwargs["doPlot"] = 0

#     nPoints = kwargs["points"].shape[0]
#     # print(f"kwargs['points'] shape: {kwargs['points'].shape}")
#     # print(f"kwargs['points'] content: {kwargs['points']}")

#     # input("###################################")
#     order = np.sum(np.sum((np.imag(kwargs["points"][:, 1:3]) == 0))) - 1    # order = np.count_nonzero(~np.isnan(kwargs["points"][:, 1:3])) - 1
    
#     xmin = np.min(kwargs["points"][:, 0])
#     xmax = np.max(kwargs["points"][:, 0])

#     # Code for most common case:
#     #   - cuvic inerpolation of 2 points
#     #       w/ function and derivative values for both
#     #   - no xminBound/xmaxBound
#     if len(kwargs) < 3:
#         kwargs["xminBound"] = xmin
#     if len(kwargs) < 4:
#         kwargs["xmaxBound"] = xmax

#     if nPoints==2 and order==3 and kwargs["doPlot"]==0:
#     # if nPoints == 2 and order == 3 and len(kwargs)<=2 and kwargs["doPlot"] == 0:
#         minVal = np.min(kwargs["points"][:, 1])
#         minPos = np.argmin(kwargs["points"][:, 1])
#         # notMinPos = -minPos + 3
#         notMinPos = 1 - minPos
#         d1 = kwargs["points"][minPos, 2] + kwargs["points"][notMinPos, 2] - 3 * (kwargs["points"][minPos, 1] - kwargs["points"][notMinPos, 1]) / (kwargs["points"][minPos, 0] - kwargs["points"][notMinPos, 0])
#         d2 = np.sqrt(d1**2 - kwargs["points"][minPos, 2] * kwargs["points"][notMinPos, 2])
#         if np.isreal(d2):
#             t = kwargs["points"][notMinPos, 0] - (kwargs["points"][notMinPos, 0] - kwargs["points"][minPos, 0]) * ((kwargs["points"][notMinPos, 2] + d2 - d1) / (kwargs["points"][notMinPos, 2] - kwargs["points"][minPos, 2] + 2 * d2))
#             minPos = np.minimum(np.maximum(t, kwargs["xminBound"]), kwargs["xmaxBound"])
#         else:
#             minPos = (kwargs["xmaxBound"]+kwargs["xminBound"])/2
#             # minPos = np.mean(kwargs["points"][:, 0])
#         # return minPos
#         # return
#         # return minPos, np.polyval(params, minPos)

#     # xmin = np.min(kwargs["points"][:, 0])
#     # xmax = np.max(kwargs["points"][:, 0])

#     # Compute Bounds of Interpolation Area
#     # if len(kwargs) < 3:
#     #     kwargs["xminBound"] = xmin
#     # if len(kwargs)<4:
#     #     kwargs["xmaxBound"] = xmax

#     # Constraints Based on available Function Values
#     A = np.zeros((0, order+1))
#     b = np.zeros((0, 1))
#     # for i in range(nPoints):
#     #     if not np.isnan(kwargs["points"][i, 1]):  # Check if function value is known (not NaN)
#     #         # Create constraint vector directly using NumPy (more efficient than loop)
#     #         constraint = np.array([kwargs["points"][i, 0] ** j for j in range(order, -1, -1)])

#     #         # Append to existing matrices (lists)
#     #         A = np.vstack((A, constraint))
#     #         b = np.vstack((b, kwargs["points"][i, 1]))
#     for i in range(nPoints):
#         if np.imag(kwargs["points"][i, 1]) == 0:
#             constraint = np.zeros((1, order + 1))
#             for j in range(order, 0, -1):
#                 constraint[order - j] = kwargs["points"][i, 0]**j
#             A = np.vstack([A, constraint])
#             b = np.vstack([b, kwargs["points"][i, 1]])


#     # Constraints based on available Derivatives
#     # for i in range(1, nPoints):
#     #     if np.isreal(kwargs["points"][i, 3]):
#     #         constraint = np.zeros((1, (order+1)))
#     #     # if kwargs["points"].shape[1] > 2 and not np.isnan(kwargs["points"][i, 2]):
#     #     #     constraint = np.zeros((1, order + 1))
#     #         for j in range(1, order):
#     #             constraint[j] = (order-j+1)*kwargs["points"][i, 1]**(order-j)
#     #         A = np.vstack((A,constraint))
#     #         b = np.vstack((b, kwargs["points"][i, 2]))
#     for i in range(nPoints):
#         if np.isreal(kwargs["points"][i, 2]):
#             constraint = np.zeros(order + 1)
#             for j in range(1, order + 1):
#                 constraint[j - 1] = (order - j + 1) * points[i, 0]**(order - j)
#             A = np.vstack([A, constraint])
#             b = np.vstack([b, kwargs["points"][i, 2]])

#     # Find interpolating polynomial
#     params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

#     # Compute Critical Points
#     dParams = np.zeros(order)
#     for i in range(len(params) - 1):
#         dParams[i] = params[i] * (order - i)

#     if np.any(np.isinf(dParams)) or np.any(np.isnan(dParams)):
#         cp = np.array([kwargs["xminBound"], kwargs["xmaxBound"], *kwargs["points"][:, 0]]) # Flatten the first column of points
#     else:
#         cp = np.array([kwargs["xminBound"], kwargs["xmaxBound"], *kwargs["points"][:, 0], *np.roots(dParams.flatten())])

#     # Test Critical Points
#     fmin = np.inf
#     minPos = (kwargs["xminBound"]+kwargs["xmaxBound"])/2

#     for xCP in cp:  # Iterate over the critical points
#         if not np.iscomplex(xCP) and kwargs["xminBound"] <= xCP <= kwargs["xmaxBound"]:  # Check if xCP is real and within bounds
#             fCP = np.polyval(params, xCP)  # Evaluate the polynomial
            
#             if np.imag(fCP)==0 and fCP<fmin:
#                 minPos = np.real(xCP)
#                 fmin = np.real(fCP)
#             # if not np.iscomplex(fCP) and fCP < fmin:  # Check if fCP is real and less than current minimum
#             #     minPos = xCP.real
#             #     fmin = fCP.real  # Convert to float to avoid potential errors with numpy types

#     # Plot Situation
    

#     if kwargs["doPlot"]:
#         plt.figure(1)
#         plt.clf()
#         # plt.hold(True)
#         plt.gca().set_prop_cycle(None)  # Reset color cycle


#         # Plot Points
#         plt.plot(kwargs["points"][:, 0], kwargs["points"][:, 1], 'b*')  # Blue star markers

#         # Plot Derivatives
#         for i in range(nPoints):
#             if not np.isnan(kwargs["points"][i, 2]):  # Equivalent of MATLAB's isreal for numeric values
#                 m = kwargs["points"][i, 2]              # Slope
#                 b = kwargs["points"][i, 1] - m * kwargs["points"][i, 0]   # Intercept
#                 x_vals = [kwargs["points"][i, 0] - 0.05, kwargs["points"][i, 0] + 0.05]  # Small interval around the point
#                 y_vals = [m * x + b for x in x_vals] # Corresponding y values using the slope and intercept
#                 plt.plot(x_vals, y_vals, 'c.-')  # Cyan line with dot markers

#         # Plot Function
#         x_min = min(kwargs["xminBound"], kwargs["xminBound"]) - 0.1
#         x_max = max(kwargs["xmaxBound"], kwargs["xmaxBound"]) + 0.1
#         x = np.linspace(x_min, x_max, 100)  # 100 points for smoother plot
#         f = np.polyval(params, x)  # Evaluate the polynomial at each x value

#         plt.plot(x, f, 'y')  # Plot the function (yellow line)

#         plt.plot(minPos, fmin, 'g+')  # Plot the minimum (green plus marker)

#         plt.axis([x[0] - 0.1, x[-1] + 0.1, min(f) - 0.1, max(f) + 0.1])  # Set axis limits with padding

#         # if kwargs["doPlot"] == 1:
#         #     plt.pause(1)  # Pause for 1 second (if desired)
#         plt.axis([np.min(x) - 0.1, np.max(x) + 0.1, np.min(f) - 0.1, np.max(f) + 0.1])  # Adjust axis limits

#         plt.show()

#         # Plot Function
#         # Create x values for plotting the function
#         x_min = min(xmin, kwargs["xminBound"]) - 0.1
#         x_max = max(xmax, kwargs["xmaxBound"]) + 0.1
#         x = np.linspace(x_min, x_max, 101)  # 101 points for smoother plot

#         # Evaluate the polynomial at each x value
#         f = np.polyval(params, x)

#         # Plot the function (yellow line)
#         plt.plot(x, f, 'y')

#         # Set axis limits with some padding
#         plt.axis([x[0] - 0.1, x[-1] + 0.1, np.min(f) - 0.1, np.max(f) + 0.1])

#         # Plot Minimum (green plus marker)
#         plt.plot(minPos, fmin, 'g+')

#         if kwargs["doPlot"] == 1:
#             plt.pause(1)  # Pause for 1 second (if desired)

#         plt.show()  # Display the plot

#     return minPos, fmin






