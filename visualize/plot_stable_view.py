# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from statsmodels.tsa import stattools


def value_stable_determinate(total_dataset):
    plt.stem(stattools.acf(total_dataset.value))
    plt.stem(stattools.pacf(total_dataset.value))
    plt.show()