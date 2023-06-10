import csv
import numpy as np


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