import numpy as np
from computeCost import *


def gradient_descent(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)
    alpha = 0.0015
    num_iters = 450 

    for i in range(0, num_iters):
        h = np.dot(X,theta)
        error = h - y
        if ( i % 50 == 0):
            print(" epoch : {i}")

        theta -= alpha*(1/m)*np.dot(X.T,error)
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        error = np.dot(X, theta) - y
        theta = theta - (alpha / m) * np.dot(X.T, error)


        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
    
