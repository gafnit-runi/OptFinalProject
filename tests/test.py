import unittest
from src.solver import solve_portfolio_optimization
from src.utils import *

STOCK_NAMES = ['AAPL', 'AMZN', 'META', 'MSFT', 'SBUX']
FILES_6M = ['../data/AAPL_6M.csv', '../data/AMZN_6M.csv', '../data/META_6M.csv', '../data/MSFT_6M.csv',
            '../data/SBUX_6M.csv']
FILES_5Y = ['../data/AAPL_5Y.csv', '../data/AMZN_5Y.csv', '../data/META_5Y.csv', '../data/MSFT_5Y.csv',
            '../data/SBUX_5Y.csv']


class TestOptimizerMethods(unittest.TestCase):

    def run_solver(self, filenames, slice_data=False, start_date=None, end_date=None):
        returns_list = []
        for filename in filenames:
            data = read_csv_file(filename)
            if slice_data:
                data = slice_data_by_date(data, start_date, end_date)
            returns = calculate_returns(data)
            returns_list.append(returns)

        covariance_matrix = calculate_covariance_matrix(returns_list)

        # Solve portfolio optimization
        weights = solve_portfolio_optimization(returns_list, covariance_matrix)
        return weights

    def test_solver_6M(self):
        """
        This test sends 6 months of data to the solver.
        :return: the optimal weights by analyzing 6 months of data.
        """
        weights = self.run_solver(FILES_6M)

        for stock, w in zip(STOCK_NAMES, weights):
            print(f'stock: {stock}, weight: {w}, check_time: 6M')

    def test_solver_5Y(self):
        """
        This test sends 5 years of data to the solver.
        :return: the optimal weights by analyzing 5 years of data.
        """
        weights = self.run_solver(FILES_5Y)

        for stock, w in zip(STOCK_NAMES, weights):
            print(f'stock: {stock}, weight: {w}, check_time: 5Y')

    def calculate_investment_return(self, weights, total_investment_amount, purchase_stocks_prices, current_stocks_prices):
        """
        Calculates a past investment return. The investment occurred at a past purchase date by investing
        total_investment_amount of money split by weights.
        :param weights: the weights for the total_investment_amount split.
        :param total_investment_amount: the total money in USD to invest.
        :param purchase_stocks_prices: the stock prices in USD at the investment date (the purchase date).
        :param current_stocks_prices: the stock prices in USD at the current date (the sell date).
        :return: the difference between the final portfolio value and the initial investment amount. The relative change
        as a fraction or percentage. Note that this can be positive, negative or zero.
        """
        # Calculate the investment amount allocated to each stock
        allocated_amounts = weights * total_investment_amount
        purchase_stocks = allocated_amounts / purchase_stocks_prices

        # Calculate the current value of each stock investment
        current_stocks_values = purchase_stocks * current_stocks_prices

        # Calculate the total portfolio value
        portfolio_value = sum(current_stocks_values)

        # Calculate the investment return
        investment_return = (portfolio_value - total_investment_amount) / total_investment_amount

        return investment_return

    def stock_prices_by_date(self, filenames, chosen_date):
        chosen_date = datetime.strptime(chosen_date, "%m/%d/%Y")
        prices_list = []
        for filename in filenames:
            data = read_csv_file(filename)
            for row in data:
                date = datetime.strptime(row['Date'], "%m/%d/%Y")
                if date == chosen_date:
                    prices_list.append(float(row['Close/Last'].replace('$', '')))
                    break
        return prices_list

    def solver_solution_return_time_window(self, start_date, end_date, current_date, investment_amount):
        """
        This function uses start_date and end_date to slice the data and run the solver on this data slice.
        :param start_date: The start date for data to analyze in the solver.
        :param end_date: The end date for data to analyze in the solver.
        :param current_date: the future date for evaluation the investment return/balance
        :param investment_amount: the total amount in USD to invest
        :return: The investment return in current date if total amount was invested by the suggestion of the solver in
        end date. The relative change as a fraction or percentage.Note that this can be positive, negative or zero.
        """

        weights = self.run_solver(FILES_5Y, slice_data=True, start_date=start_date, end_date=end_date)
        for stock, w in zip(STOCK_NAMES, weights):
            print(f'stock: {stock}, weight: {w}, check_time: {start_date} - {end_date}')

        purchase_stocks_prices = self.stock_prices_by_date(FILES_5Y, end_date)
        current_stocks_prices = self.stock_prices_by_date(FILES_5Y, current_date)
        investment_return = self.calculate_investment_return(weights, investment_amount, purchase_stocks_prices, current_stocks_prices)

        return investment_return

    def test_solver_solution(self):
        """
        This test runs multiple checks of past solver suggestions and checks if the solution was optimal based on the
        future.
        Dates are in the format of "%m/%d/%Y"
        :return:
        """
        investment_amount = 10000
        investment_return = self.solver_solution_return_time_window(start_date='12/12/2022', end_date='04/12/2023', current_date='06/09/2023', investment_amount=investment_amount)

        print(f'investment_return: {investment_return}')

        investment_return = self.solver_solution_return_time_window(start_date='01/01/2020', end_date='01/06/2020', current_date='06/09/2023', investment_amount=investment_amount)

        print(f'investment_return: {investment_return}')


if __name__ == '__main__':
    unittest.main()
