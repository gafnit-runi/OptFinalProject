from cvxopt import matrix, solvers
import numpy as np


def solve_portfolio_optimization(returns, covariance_matrix):
    num_assets = len(returns)

    # Convert data to cvxopt matrices
    Q = matrix(covariance_matrix)
    r = matrix(np.zeros(num_assets))
    A = matrix(np.ones(num_assets)).T
    b = matrix(1.0)
    G = matrix(- np.eye(num_assets))
    h = matrix(np.zeros(num_assets))

    # Solve the quadratic program
    sol = solvers.qp(Q, r, G, h, A, b)
    if sol['status'] == 'optimal':
        weights = np.array(sol['x']).flatten()
        return weights
    else:
        raise ValueError("Failed to find optimal solution.")