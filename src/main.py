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
    return np.cov(returns, rowvar=False)


def solve_portfolio_optimization(returns, covariance_matrix, target_return):
    pass


if __name__ == '__main__':
    # List of filenames for the CSV files
    filenames = ['../data/AAPL_6M.csv', '../data/AMZN_6M.csv', '../data/META_6M.csv', '../data/MSFT_6M.csv', '../data/SBUX_6M.csv']

    # Read data from CSV files and calculate mean and variance for each stock
    returns_list = []
    for filename in filenames:
        data = read_csv_file(filename)
        returns = calculate_returns(data)
        returns_list.append(returns)

    covariance_matrix = calculate_covariance_matrix(returns_list)
    target_return = 0.05  # Set your desired target return here

    # Solve portfolio optimization
    weights = solve_portfolio_optimization(returns_list, covariance_matrix, target_return)

    print("Optimal Weights:", weights)
