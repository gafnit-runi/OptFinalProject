from solver import solve_portfolio_optimization
from utils import *

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
