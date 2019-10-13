# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from statsmodels.tsa import stattools


def value_stable_determinate_view(total_dataset):
    """

    :param total_dataset: the dataset that contains analysis data
    :return: a plot to determine whether the time series is stable
    """
    plt.stem(stattools.acf(total_dataset.value))
    plt.stem(stattools.pacf(total_dataset.value))
    plt.show()