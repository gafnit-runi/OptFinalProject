import csv
import numpy as np
from cvxopt import matrix, solvers


def read_csv_file(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data


def calculate_returns(data):
    close_prices = [float(row['Close/Last'].replace('$', '')) for row in data]
    returns = np.diff(close_prices) / close_prices[:-1]
    return returns


def calculate_covariance_matrix(returns):
    return np.cov(returns)


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


if __name__ == '__main__':
    stock_names = ['AAPL', 'AMZN', 'META', 'MSFT', 'SBUX']
    # List of filenames for the CSV files
    filenames = ['../data/AAPL_6M.csv', '../data/AMZN_6M.csv', '../data/META_6M.csv', '../data/MSFT_6M.csv',
                 '../data/SBUX_6M.csv']

    # Read data from CSV files and calculate mean and variance for each stock
    returns_list = []
    for filename in filenames:
        data = read_csv_file(filename)
        returns = calculate_returns(data)
        returns_list.append(returns)

    covariance_matrix = calculate_covariance_matrix(returns_list)

    # Solve portfolio optimization
    weights = solve_portfolio_optimization(returns_list, covariance_matrix)

    for stock, w in zip(stock_names, weights):
        print(f'stock: {stock}, weight: {w}')
