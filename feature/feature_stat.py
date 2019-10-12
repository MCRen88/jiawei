#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Tencent is pleased to support the open source community by making Metis available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""
from __future__ import absolute_import, division
import pandas as pd
import numpy as np
import warnings
import tsfresh.feature_extraction.feature_calculators as ts_feature_calculators
from builtins import range



# todo: make sure '_' works in parameter names in all cases, add a warning if not



__all__ = [ "time_series_maximum",
            "time_series_minimum",
            "time_series_mean",
            "time_series_variance",
            "time_series_standard_deviation",
            "time_series_skewness",
            "time_series_kurtosis",
            "time_series_median",
            "time_series_sum_values",
            "time_series_abs_energy",
            "time_series_absolute_sum_of_changes",
            "time_series_variance_larger_than_std",
            "time_series_count_above_mean",
            "time_series_count_below_mean",
            "time_series_mean_abs_change",
            "time_series_percentage_of_reoccurring_datapoints_to_all_datapoints",
            "time_series_ratio_value_number_to_time_series_length",
            "time_series_sum_of_reoccurring_data_points",
            "time_series_sum_of_reoccurring_values",
            "time_series_range",
            "time_series_mean_change",
            "time_series_has_duplicate",
            "time_series_has_duplicate_max",
            "time_series_has_duplicate_min",
            "time_series_longest_strike_above_mean",
            "time_series_longest_strike_below_mean",
            "abs_energy",
            "friedrich_coefficients",
            "ratio_beyond_r_sigma",
            "large_standard_deviation",
            "number_peaks",
            "fft_aggregated", ##unsure
            "ratio_beyond_r_sigma"
            ]



def _roll(a, shift):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])

def time_series_maximum(x):
    """
    Calculates the highest value of the time series x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.maximum(x)


def time_series_minimum(x):
    """
    Calculates the lowest value of the time series x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.minimum(x)


def time_series_mean(x):
    """
    Returns the mean of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.mean(x)


def time_series_variance(x):
    """
    Returns the variance of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.variance(x)


def time_series_standard_deviation(x):
    """
    Returns the standard deviation of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.standard_deviation(x)


def time_series_skewness(x):
    """
    Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1).
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.skewness(x)


def time_series_kurtosis(x):
    """
    Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2).
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.kurtosis(x)


def time_series_median(x):
    """
    Returns the median of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.median(x)


def time_series_abs_energy(x):
    """
    Returns the absolute energy of the time series which is the sum over the squared values
    .. math::
        E = \\sum_{i=1,\ldots, n} x_i^2
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.abs_energy(x)


def time_series_absolute_sum_of_changes(x):
    """
    Returns the sum over the absolute value of consecutive changes in the series x
    .. math::
        \\sum_{i=1, \ldots, n-1} \\mid x_{i+1}- x_i \\mid
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.absolute_sum_of_changes(x)


def time_series_variance_larger_than_std(x):
    """
    Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
    being larger than 1
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: int
    """
    return int(ts_feature_calculators.variance_larger_than_standard_deviation(x))


def time_series_count_above_mean(x):
    """
    Returns the number of values in x that are higher than the mean of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.count_above_mean(x)


def time_series_count_below_mean(x):
    """
    Returns the number of values in x that are lower than the mean of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.count_below_mean(x)


def time_series_first_location_of_maximum(x):
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.first_location_of_maximum(x)


def time_series_first_location_of_minimum(x):
    """
    Returns the first location of the minimal value of x.
    The position is calculated relatively to the length of x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.first_location_of_minimum(x)


def time_series_last_location_of_maximum(x):
    """
    Returns the relative last location of the maximum value of x.
    The position is calculated relatively to the length of x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.last_location_of_maximum(x)


def time_series_last_location_of_minimum(x):
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.last_location_of_minimum(x)


def time_series_has_duplicate(x):
    """
    Checks if any value in x occurs more than once
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return ts_feature_calculators.has_duplicate(x)


def time_series_has_duplicate_max(x):
    """
    Checks if the maximum value of x is observed more than once
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return ts_feature_calculators.has_duplicate_max(x)


def time_series_has_duplicate_min(x):
    """
    Checks if the minimal value of x is observed more than once
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return ts_feature_calculators.has_duplicate_min(x)


def time_series_longest_strike_above_mean(x):
    """
    Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.longest_strike_above_mean(x)


def time_series_longest_strike_below_mean(x):
    """
    Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.longest_strike_below_mean(x)


def time_series_mean_abs_change(x):
    """
    Returns the mean over the absolute differences between subsequent time series values which is
    .. math::
        \\frac{1}{n} \\sum_{i=1,\ldots, n-1} | x_{i+1} - x_{i}|
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.mean_abs_change(x)


def time_series_mean_change(x):
    """
    Returns the mean over the absolute differences between subsequent time series values which is
    .. math::
        \\frac{1}{n} \\sum_{i=1,\ldots, n-1}  x_{i+1} - x_{i}
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.mean_change(x)


def time_series_percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    """
    Returns the percentage of unique values, that are present in the time series
    more than once.
        len(different values occurring more than once) / len(different values)
    This means the percentage is normalized to the number of unique values,
    in contrast to the percentage_of_reoccurring_values_to_all_values.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.percentage_of_reoccurring_datapoints_to_all_datapoints(x)


def time_series_ratio_value_number_to_time_series_length(x):
    """
    Returns a factor which is 1 if all values in the time series occur only once,
    and below one if this is not the case.
    In principle, it just returns
        # unique values / # values
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.ratio_value_number_to_time_series_length(x)


def time_series_sum_of_reoccurring_data_points(x):
    """
    Returns the sum of all data points, that are present in the time series
    more than once.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.sum_of_reoccurring_data_points(x)


def time_series_sum_of_reoccurring_values(x):
    """
    Returns the sum of all values, that are present in the time series
    more than once.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.sum_of_reoccurring_values(x)


def time_series_sum_values(x):
    """
    Calculates the sum over the time series values
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return ts_feature_calculators.sum_values(x)


def time_series_range(x):
    """
    Calculates the range value of the time series x.
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return time_series_maximum(x) - time_series_minimum(x)

# add yourself statistical features here...


def abs_energy(x):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)



def friedrich_coefficients(x, param):
    """
    Coefficients of polynomial :math:`h(x)`, which has been fitted to
    the deterministic dynamics of Langevin model

    .. math::
        \dot{x}(t) = h(x(t)) + \mathcal{N}(0,R)

    as described by [1].

    For short time-series this method is highly dependent on the parameters.

    .. rubric:: References

    |  [1] Friedrich et al. (2000): Physics Letters A 271, p. 217-222
    |  *Extracting model equations from experimental data*

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"m": x, "r": y, "coeff": z} with x being positive integer, the order of polynom to fit for estimating fixed points of
                    dynamics, y positive float, the number of quantils to use for averaging and finally z, a positive integer corresponding to the returned
                    coefficient
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    calculated = {}  # calculated is dictionary storing the calculated coefficients {m: {r: friedrich_coefficients}}
    res = {}  # res is a dictionary containg the results {"m_10__r_2__coeff_3": 15.43}

    for parameter_combination in param:
        m = parameter_combination['m']
        r = parameter_combination['r']
        coeff = parameter_combination["coeff"]

        assert coeff >= 0, "Coefficients must be positive or zero. Found {}".format(coeff)

        # calculate the current friedrich coefficients if they do not exist yet
        if m not in calculated:
            calculated[m] = {r: _estimate_friedrich_coefficients(x, m, r)}
        else:
            if r not in calculated[m]:
                calculated[m] = {r: _estimate_friedrich_coefficients(x, m, r)}

        try:
            res["m_{}__r_{}__coeff_{}".format(m, r, coeff)] = calculated[m][r][coeff]
        except IndexError:
            res["m_{}__r_{}__coeff_{}".format(m, r, coeff)] = np.NaN
        # return [(key, value) for key, value in res.items()]
        return [value for key, value in res.items()]


def ratio_beyond_r_sigma(x, r):
    """
    Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.

    :param x: the time series to calculate the feature of
    :type x: iterable
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.sum(np.abs(x - np.mean(x)) > r * np.std(x))/x.size


def large_standard_deviation(x, r):
    """
    Boolean variable denoting if the standard dev of x is higher
    than 'r' times the range = difference between max and min of x.
    Hence it checks if

    .. math::

        std(x) > r * (max(X)-min(X))

    According to a rule of the thumb, the standard deviation should be a forth of the range of the values.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param r: the percentage of the range to compare with
    :type r: float
    :return: the value of this feature
    :return type: bool
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.std(x) > (r * (np.max(x) - np.min(x)))

def number_peaks(x, n):
    """
    Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
    subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

    Hence in the sequence

    # >>> x = [3, 0, 0, 4, 0, 0, 13]

    4 is a peak of support 1 and 2 because in the subsequences

    # >>> [0, 4, 0]
    # >>> [0, 0, 4, 0, 0]

    4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
    and its bigger than 4.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param n: the support of the peak
    :type n: int
    :return: the value of this feature
    :return type: float
    """
    x_reduced = x[n:-n]

    res = None
    for i in range(1, n + 1):
        result_first = (x_reduced > _roll(x, i)[n:-n])

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= (x_reduced > _roll(x, -i)[n:-n])
    return np.sum(res)


def fft_aggregated(x, param):
    """
    Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"aggtype": s} where s str and in ["centroid", "variance",
        "skew", "kurtosis"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    assert set([config["aggtype"] for config in param]) <= set(["centroid", "variance", "skew", "kurtosis"]), \
        'Attribute must be "centroid", "variance", "skew", "kurtosis"'


def ratio_beyond_r_sigma(x, r):
    """
    Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.

    :param x: the time series to calculate the feature of
    :type x: iterable
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.sum(np.abs(x - np.mean(x)) > r * np.std(x))/x.size