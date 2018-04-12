from covariance_matrix.covariance_matrix import calc_covariance, calc_variance
import math
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def correlation(v1, v2):
    assert len(v1) == len(v2)
    covariance = calc_covariance(v1, v2)
    standard_deviation_1 = math.sqrt(calc_variance(v1))
    standard_deviation_2 = math.sqrt(calc_variance(v2))
    return covariance / (standard_deviation_1 * standard_deviation_2)


if __name__ == '__main__':
    n = 100
    x = [i * 0.1 for i in range(-n, n)]
    y = [x[i] * x[i] + random.random() * 10 for i in range(0, len(x))]

    data = pd.DataFrame({'x': x, 'y': y})

    sns.lmplot('x', 'y', data=data, fit_reg=False,
               legend=False, scatter_kws={"s": 10},
               palette='gist_rainbow')

    plt.show()
    print(np.corrcoef(x, y))
    print(correlation(x, y))
