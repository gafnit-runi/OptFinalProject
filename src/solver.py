from cvxopt import matrix, solvers
import numpy as np


def solve_portfolio_optimization(returns, G, mean_vec, k=0.5):
    n = len(returns)

    # Convert data to cvxopt matrices
    P = matrix(2 * k * G)
    q = matrix(mean_vec)
    A = matrix(np.ones(n)).T
    b = matrix(1.0)
    B = matrix(- np.eye(n))
    h = matrix(np.zeros(n))

    # Solve the quadratic program
    sol = solvers.qp(P=P, q=q, G=B, h=h, A=A, b=b)
    if sol['status'] == 'optimal':
        weights = np.array(sol['x']).flatten()
        return weights
    else:
        raise ValueError("Failed to find optimal solution.")