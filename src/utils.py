import csv
import numpy as np
from datetime import datetime


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


def slice_data_by_date(data, start_date, end_date):
    start_date = datetime.strptime(start_date, "%m/%d/%Y")
    end_date = datetime.strptime(end_date, "%m/%d/%Y")

    sliced_data = []
    for row in data:
        date = datetime.strptime(row['Date'], "%m/%d/%Y")
        if start_date <= date <= end_date:
            sliced_data.append(row)

    return sliced_data
