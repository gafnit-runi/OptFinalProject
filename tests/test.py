import unittest
from datetime import datetime, timedelta

import numpy as np

from src.solver import solve_portfolio_optimization
from src.utils import *
import matplotlib.pyplot as plt
import os
from datetime import datetime

STOCK_NAMES = set()
FILES_6M = set()
FILES_5Y = set()
for filename in os.listdir('../data'):
    STOCK_NAMES.add(filename.split('_')[0])
    if '6M' in filename:
        FILES_6M.add(f'../data/{filename}')
    if '5Y' in filename:
        FILES_5Y.add(f'../data/{filename}')

STOCK_NAMES = list(STOCK_NAMES)
FILES_6M = list(FILES_6M)
FILES_5Y = list(FILES_5Y)

STOCKS_LATEST_DATE = '06/22/2023'
STOCKS_OLDEST_DATE = '06/25/2018'

DATE_FORMAT = '%m/%d/%Y'

STOCK_PRICES_BY_DATE = {}

start_date = datetime.strptime(STOCKS_LATEST_DATE, '%m/%d/%Y')
end_date = datetime.strptime(STOCKS_OLDEST_DATE, '%m/%d/%Y')

current_date = start_date
while current_date >= end_date:
    STOCK_PRICES_BY_DATE[current_date.strftime('%m/%d/%Y')] = []
    current_date -= timedelta(days=1)

for filename in FILES_5Y:
    stock_prices_by_date = {key: None for key in STOCK_PRICES_BY_DATE.keys()}
    data = read_csv_file(filename)
    for row in data:
        date = row['Date']
        stock_prices_by_date[date] = float(row['Close/Last'].replace('$', ''))

    for d, p in stock_prices_by_date.items():
        if not p:
            prev_date = (datetime.strptime(d, '%m/%d/%Y') + timedelta(days=1)).strftime('%m/%d/%Y')
            stock_prices_by_date[d] = stock_prices_by_date[prev_date]
        STOCK_PRICES_BY_DATE[d].append(stock_prices_by_date[d])


class TestOptimizerMethods(unittest.TestCase):

    def run_solver(self, filenames, slice_data=False, start_date=None, end_date=None):
        returns_list = []
        for filename in filenames:
            data = read_csv_file(filename)
            if slice_data:
                data = slice_data_by_date(data, start_date, end_date)
                if not data:
                    print(filename)
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

    def calculate_investment_return(self, weights, total_investment_amount, purchase_stocks_prices,
                                    sale_stocks_prices):
        """
        Calculates a past investment return. The investment occurred at a past purchase date by investing
        total_investment_amount of money split by weights.
        :param weights: the weights for the total_investment_amount split.
        :param total_investment_amount: the total money in USD to invest.
        :param purchase_stocks_prices: the stock prices in USD at the investment date (the purchase date).
        :param sale_stocks_prices: the stock prices in USD at the current date (the sell date).
        :return: the difference between the final portfolio value and the initial investment amount. The relative change
        as a fraction or percentage. Note that this can be positive, negative or zero.
        """
        # Calculate the investment amount allocated to each stock
        allocated_amounts = weights * total_investment_amount
        purchase_stocks = allocated_amounts / purchase_stocks_prices

        # Calculate the current value of each stock investment
        current_stocks_values = purchase_stocks * sale_stocks_prices

        # Calculate the total portfolio value
        portfolio_value = sum(current_stocks_values)

        # Calculate the investment return
        investment_return = (portfolio_value - total_investment_amount) / total_investment_amount

        return investment_return

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

        purchase_stocks_prices = STOCK_PRICES_BY_DATE[end_date]
        current_stocks_prices = STOCK_PRICES_BY_DATE[current_date]
        investment_return = self.calculate_investment_return(weights, investment_amount, purchase_stocks_prices,
                                                             current_stocks_prices)

        return investment_return

    def get_investment_returns(self, weights, invest_date, current_date, investment_amount):

        investment_returns = []
        sale_date = datetime.strptime(invest_date, DATE_FORMAT) + timedelta(days=1)
        purchase_stocks_prices = STOCK_PRICES_BY_DATE[invest_date]
        current_date = datetime.strptime(current_date, DATE_FORMAT)

        while sale_date <= current_date:
            sale_stocks_prices = STOCK_PRICES_BY_DATE[sale_date.strftime(DATE_FORMAT)]
            investment_return = self.calculate_investment_return(weights, investment_amount, purchase_stocks_prices,
                                                                 sale_stocks_prices)
            investment_returns.append(investment_return)
            sale_date = sale_date + timedelta(days=1)

        mean = np.mean(investment_returns)
        variance = np.var(investment_returns)

        return investment_returns, mean, variance

    def test_solver_solution(self):
        """
        This test runs multiple checks of past solver suggestions and checks if the solution was optimal based on the
        future.
        Dates are in the format of "%m/%d/%Y"
        :return:
        """
        investment_amount = 10000
        investment_return = self.solver_solution_return_time_window(start_date='12/12/2022', end_date='04/12/2023',
                                                                    current_date='06/09/2023',
                                                                    investment_amount=investment_amount)

        print(f'investment_return: {investment_return}')

        investment_return = self.solver_solution_return_time_window(start_date='01/01/2020', end_date='01/06/2020',
                                                                    current_date='06/09/2023',
                                                                    investment_amount=investment_amount)

        print(f'investment_return: {investment_return}')

    def test_solver_by_window_size(self):
        """
        This test runs the solver with multiple window sizes. In this test we assume that we want to invest for at
        least 1 year. We mark the date 1 year ago as the investment date and start to increase the solver window
        relatively to the investment date. Next, we check each window solver weights by calculating the investment
        return mean and variance. We compare it with uniform and random investments at the investment date.
        :return: plotting 2 plots, one for the investment mean and one for the variance. Each plot has 3 graphs: solver,
        uniform and random.
        """
        # invest at least for 1 year
        invest_date = datetime.strptime(STOCKS_LATEST_DATE, DATE_FORMAT) - timedelta(days=365)
        investment_amount = 10000
        window_size_days = 10

        window_sizes = []
        investment_returns_solver_mean = []
        investment_returns_uniform_mean = []
        investment_returns_random_mean = []
        investment_returns_solver_variance = []
        investment_returns_uniform_variance = []
        investment_returns_random_variance = []

        while invest_date - timedelta(days=window_size_days) >= datetime.strptime(STOCKS_OLDEST_DATE, DATE_FORMAT):
            window_sizes.append(window_size_days)
            window_start = invest_date - timedelta(days=window_size_days)
            window_start_str = window_start.strftime(DATE_FORMAT)
            window_end_str = invest_date.strftime(DATE_FORMAT)

            # solver
            solver_weights = self.run_solver(FILES_5Y, slice_data=True, start_date=window_start_str,
                                             end_date=window_end_str)
            _, solver_mean, solver_variance = self.get_investment_returns(solver_weights, invest_date=window_end_str,
                                                                          current_date=STOCKS_LATEST_DATE,
                                                                          investment_amount=investment_amount)
            investment_returns_solver_mean.append(solver_mean)
            investment_returns_solver_variance.append(solver_variance)

            # uniform
            uniform_weights = np.ones(len(STOCK_NAMES)) / len(STOCK_NAMES)
            _, uniform_mean, uniform_variance = self.get_investment_returns(uniform_weights, invest_date=window_end_str,
                                                                            current_date=STOCKS_LATEST_DATE,
                                                                            investment_amount=investment_amount)
            investment_returns_uniform_mean.append(uniform_mean)
            investment_returns_uniform_variance.append(uniform_variance)

            # random
            random_weights = np.random.dirichlet(np.ones(len(STOCK_NAMES)), size=1)[0]
            _, random_mean, random_variance = self.get_investment_returns(random_weights, invest_date=window_end_str,
                                                                          current_date=STOCKS_LATEST_DATE,
                                                                          investment_amount=investment_amount)
            investment_returns_random_mean.append(random_mean)
            investment_returns_random_variance.append(random_variance)

            window_size_days += 50

        # self.plot_weights_dates_results(weights_by_window_size, window_type='size')
        plt.figure(1)
        plt.plot(window_sizes, investment_returns_solver_mean, label='Solver Mean', marker='o')
        plt.plot(window_sizes, investment_returns_uniform_mean, label='Uniform Mean', marker='o')
        plt.plot(window_sizes, investment_returns_random_mean, label='Random Mean', marker='o')

        plt.xlabel('window size (days)')
        plt.ylabel('investment return mean')
        plt.title('Investment Returns Mean vs. Window Sizes')
        plt.legend()

        plt.figure(2)
        plt.plot(window_sizes, investment_returns_solver_variance, label='Solver Variance', marker='o')
        plt.plot(window_sizes, investment_returns_uniform_variance, label='Uniform Variance', marker='o')
        plt.plot(window_sizes, investment_returns_random_variance, label='Random Variance', marker='o')

        plt.xlabel('window size (days)')
        plt.ylabel('investment return variance')
        plt.title('Investment Returns Variance vs. Window Sizes')
        plt.legend()

        plt.show()

    def test_solver_by_window_location(self):
        """
        This test runs the solver with multiple window location. In all test we are interested to give the solver a
        window of 860 days (based on the window test), and to invest for 1 year. In each test we move the solver
        window by changing the investment date. Next, we check each window solver weights by calculating the
        investment return mean and variance. We compare it with uniform and random investments at the investment
        date.
        :return: plotting 2 plots, one for the investment mean and one for the variance. Each plot has 3
        graphs: solver, uniform and random.
        """
        # invest at least for 1 year
        invest_date = datetime.strptime(STOCKS_LATEST_DATE, DATE_FORMAT) - timedelta(days=365)
        investment_amount = 10000
        window_size_days = 860

        window_invest_locations = []
        investment_returns_solver_mean = []
        investment_returns_uniform_mean = []
        investment_returns_random_mean = []
        investment_returns_solver_variance = []
        investment_returns_uniform_variance = []
        investment_returns_random_variance = []

        while invest_date - timedelta(days=window_size_days) >= datetime.strptime(STOCKS_OLDEST_DATE, DATE_FORMAT):
            window_invest_locations.append(invest_date.strftime(DATE_FORMAT))
            window_start = invest_date - timedelta(days=window_size_days)
            window_start_str = window_start.strftime(DATE_FORMAT)
            window_end_str = invest_date.strftime(DATE_FORMAT)

            # solver
            solver_weights = self.run_solver(FILES_5Y, slice_data=True, start_date=window_start_str,
                                             end_date=window_end_str)
            _, solver_mean, solver_variance = self.get_investment_returns(solver_weights, invest_date=window_end_str,
                                                                          current_date=STOCKS_LATEST_DATE,
                                                                          investment_amount=investment_amount)
            investment_returns_solver_mean.append(solver_mean)
            investment_returns_solver_variance.append(solver_variance)

            # uniform
            uniform_weights = np.ones(len(STOCK_NAMES)) / len(STOCK_NAMES)
            _, uniform_mean, uniform_variance = self.get_investment_returns(uniform_weights, invest_date=window_end_str,
                                                                            current_date=STOCKS_LATEST_DATE,
                                                                            investment_amount=investment_amount)
            investment_returns_uniform_mean.append(uniform_mean)
            investment_returns_uniform_variance.append(uniform_variance)

            # random
            random_weights = np.random.dirichlet(np.ones(len(STOCK_NAMES)), size=1)[0]
            _, random_mean, random_variance = self.get_investment_returns(random_weights, invest_date=window_end_str,
                                                                          current_date=STOCKS_LATEST_DATE,
                                                                          investment_amount=investment_amount)
            investment_returns_random_mean.append(random_mean)
            investment_returns_random_variance.append(random_variance)

            invest_date -= timedelta(days=50)

        # self.plot_weights_dates_results(weights_by_window_size, window_type='size')
        plt.figure(1)
        plt.plot(window_invest_locations[::-1], investment_returns_solver_mean[::-1], label='Solver Mean', marker='o')
        plt.plot(window_invest_locations[::-1], investment_returns_uniform_mean[::-1], label='Uniform Mean', marker='o')
        plt.plot(window_invest_locations[::-1], investment_returns_random_mean[::-1], label='Random Mean', marker='o')

        plt.xlabel('window invest date')
        plt.ylabel('investment return mean')
        plt.xticks(rotation=45, ha='right')
        plt.title('Investment Returns Mean vs. Window Location\n1 year invest, 860 days solver window')
        plt.legend()

        plt.figure(2)
        plt.plot(window_invest_locations[::-1], investment_returns_solver_variance[::-1], label='Solver Variance',
                 marker='o')
        plt.plot(window_invest_locations[::-1], investment_returns_uniform_variance[::-1], label='Uniform Variance',
                 marker='o')
        plt.plot(window_invest_locations[::-1], investment_returns_random_variance[::-1], label='Random Variance',
                 marker='o')

        plt.xlabel('window invest date')
        plt.ylabel('investment return variance')
        plt.xticks(rotation=45, ha='right')
        plt.title('Investment Returns Variance vs. Window Location\n1 year invest, 860 days solver window')
        plt.legend()

        plt.show()

    def test_solver_investment_return(self):
        """
        This test runs the solver with a window size of 860 days and invest date 06/22/2022 (based on previous window
        size and location tests). In this test we assume that we want to invest for at least 1 year. We mark the date
        1 year ago as the investment date. Next, compare the investment return over time along the year. We compare
        it with uniform and random investments at the investment date.
        :return: plot the investment return over time
        along the year for solver, uniform and random.
        """
        # invest at least for 1 year
        invest_date = datetime.strptime(STOCKS_LATEST_DATE, DATE_FORMAT) - timedelta(days=365)
        investment_amount = 10000
        window_size_days = 860

        window_start = invest_date - timedelta(days=window_size_days)
        window_start_str = window_start.strftime(DATE_FORMAT)
        window_end_str = invest_date.strftime(DATE_FORMAT)

        # solver
        solver_weights = self.run_solver(FILES_5Y, slice_data=True, start_date=window_start_str,
                                         end_date=window_end_str)
        solver_returns, _, _ = self.get_investment_returns(solver_weights, invest_date=window_end_str,
                                                           current_date=STOCKS_LATEST_DATE,
                                                           investment_amount=investment_amount)

        # uniform
        uniform_weights = np.ones(len(STOCK_NAMES)) / len(STOCK_NAMES)
        uniform_returns, _, _ = self.get_investment_returns(uniform_weights, invest_date=window_end_str,
                                                            current_date=STOCKS_LATEST_DATE,
                                                            investment_amount=investment_amount)

        # random
        random_weights = np.random.dirichlet(np.ones(len(STOCK_NAMES)), size=1)[0]
        random_returns, _, _ = self.get_investment_returns(random_weights, invest_date=window_end_str,
                                                           current_date=STOCKS_LATEST_DATE,
                                                           investment_amount=investment_amount)

        dates = []
        current_date = invest_date + timedelta(days=1)
        while current_date <= datetime.strptime(STOCKS_LATEST_DATE, DATE_FORMAT):
            dates.append(current_date.strftime('%m/%d/%Y'))
            current_date += timedelta(days=1)

        plt.plot(dates, solver_returns, label='Solver')
        plt.plot(dates, uniform_returns, label='Uniform')
        plt.plot(dates, random_returns, label='Random')

        plt.xlabel('date')
        plt.ylabel('investment return')
        plt.title('Investment Returns by Date\n1 year invest, 860 days solver window')
        plt.legend()

        x_ticks = np.arange(0, len(dates), 50)
        x_labels = [dates[i] for i in x_ticks]

        plt.xticks(x_ticks, x_labels, rotation=45)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
