import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def calc_variance(x):
    mean = calc_mean(x)
    return sum([(item - mean) ** 2 for item in x]) / len(x)


def calc_mean(x):
    return sum(x) / len(x)


def calc_covariance(x, y):
    assert len(x) == len(y)
    n = len(x)
    x_mean = calc_mean(x)
    y_mean = calc_mean(y)
    return sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(0, n)]) / (
                n - 1)


def calc_covariance_matrix(x):
    n = len(x)
    cov_matrix = [[0] * n for _ in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            cov_matrix[i][j] = calc_covariance(x[i], x[j])
    return cov_matrix


def main():
    n = 100
    x = [i * 0.1 for i in range(0, n)]
    y = [x[i] + random.random() * 10 for i in range(0, n)]

    data = pd.DataFrame({'x': x, 'y': y})

    sns.lmplot('x', 'y', data=data, fit_reg=False,
               legend=False, scatter_kws={"s": 10},
               palette='gist_rainbow')

    plt.show()
    print(np.cov([x, y]))
    print(calc_covariance_matrix([x, y]))


if __name__ == '__main__':
    main()
