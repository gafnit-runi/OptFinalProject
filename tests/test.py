import unittest
from src.solver import solve_portfolio_optimization
from src.utils import *

STOCK_NAMES = ['AAPL', 'AMZN', 'META', 'MSFT', 'SBUX']
FILES_6M = ['../data/AAPL_6M.csv', '../data/AMZN_6M.csv', '../data/META_6M.csv', '../data/MSFT_6M.csv',
            '../data/SBUX_6M.csv']
FILES_5Y = ['../data/AAPL_5Y.csv', '../data/AMZN_5Y.csv', '../data/META_5Y.csv', '../data/MSFT_5Y.csv',
            '../data/SBUX_5Y.csv']


class TestOptimizerMethods(unittest.TestCase):

    def run_solver(self, filenames):
        returns_list = []
        for filename in filenames:
            data = read_csv_file(filename)
            returns = calculate_returns(data)
            returns_list.append(returns)

        covariance_matrix = calculate_covariance_matrix(returns_list)

        # Solve portfolio optimization
        weights = solve_portfolio_optimization(returns_list, covariance_matrix)
        return weights

    def test_solver_6M(self):
        weights = self.run_solver(FILES_6M)

        for stock, w in zip(STOCK_NAMES, weights):
            print(f'stock: {stock}, weight: {w}, check_time: 6M')

    def test_solver_5Y(self):
        weights = self.run_solver(FILES_5Y)

        for stock, w in zip(STOCK_NAMES, weights):
            print(f'stock: {stock}, weight: {w}, check_time: 5Y')


if __name__ == '__main__':
    unittest.main()
