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
            "time_series_first_location_of_maximum",
            "time_series_first_location_of_minimum",
            "time_series_last_location_of_maximum",
            "time_series_last_location_of_minimum",
            "abs_energy",
            "friedrich_coefficients",
            "ratio_beyond_r_sigma",
            "large_standard_deviation",
            "number_peaks",
            "fft_aggregated" ##unsure
            ]



def _roll(a, shift):
    """
    :param a: the input array
    :type a: array_like
    :param shift: the number of places by which elements are shifted
    :type shift: int

    :return: shifted array with the same shape as a
    :return type: ndarray
    """
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])

def time_series_maximum(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.maximum(x)


def time_series_minimum(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.minimum(x)


def time_series_mean(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.mean(x)


def time_series_variance(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.variance(x)


def time_series_standard_deviation(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.standard_deviation(x)


def time_series_skewness(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.skewness(x)


def time_series_kurtosis(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.kurtosis(x)


def time_series_median(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.median(x)


def time_series_abs_energy(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.abs_energy(x)


def time_series_absolute_sum_of_changes(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.absolute_sum_of_changes(x)


def time_series_variance_larger_than_std(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: int
    """
    return int(ts_feature_calculators.variance_larger_than_standard_deviation(x))


def time_series_count_above_mean(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.count_above_mean(x)


def time_series_count_below_mean(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.count_below_mean(x)


def time_series_first_location_of_maximum(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.first_location_of_maximum(x)


def time_series_first_location_of_minimum(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.first_location_of_minimum(x)


def time_series_last_location_of_maximum(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.last_location_of_maximum(x)


def time_series_last_location_of_minimum(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.last_location_of_minimum(x)


def time_series_has_duplicate(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return ts_feature_calculators.has_duplicate(x)


def time_series_has_duplicate_max(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return ts_feature_calculators.has_duplicate_max(x)


def time_series_has_duplicate_min(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return ts_feature_calculators.has_duplicate_min(x)


def time_series_longest_strike_above_mean(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.longest_strike_above_mean(x)


def time_series_longest_strike_below_mean(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.longest_strike_below_mean(x)


def time_series_mean_abs_change(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.mean_abs_change(x)


def time_series_mean_change(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.mean_change(x)


def time_series_percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.percentage_of_reoccurring_datapoints_to_all_datapoints(x)


def time_series_ratio_value_number_to_time_series_length(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.ratio_value_number_to_time_series_length(x)


def time_series_sum_of_reoccurring_data_points(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.sum_of_reoccurring_data_points(x)


def time_series_sum_of_reoccurring_values(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return ts_feature_calculators.sum_of_reoccurring_values(x)


def time_series_sum_values(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return ts_feature_calculators.sum_values(x)


def time_series_range(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    return time_series_maximum(x) - time_series_minimum(x)



def abs_energy(x):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)


def _estimate_friedrich_coefficients(x, m, r):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param m: order of polynom to fit for estimating fixed points of dynamics
    :type m: int
    :param r: number of quantils to use for averaging
    :type r: float

    :return: coefficients of polynomial of deterministic dynamics
    :return type: ndarray
    """
    assert m > 0, "Order of polynomial need to be positive integer, found {}".format(m)

    df = pd.DataFrame({'signal': x[:-1], 'delta': np.diff(x)})
    try:
        df['quantiles'] = pd.qcut(df.signal, r)
    except ValueError:
        return [np.NaN] * (m + 1)

    quantiles = df.groupby('quantiles')

    result = pd.DataFrame({'x_mean': quantiles.signal.mean(), 'y_mean': quantiles.delta.mean()})
    result.dropna(inplace=True)

    try:
        return np.polyfit(result.x_mean, result.y_mean, deg=m)
    except (np.linalg.LinAlgError, ValueError):
        return [np.NaN] * (m + 1)


def friedrich_coefficients(x, param):
    """
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
        return [value for key, value in res.items()]


def ratio_beyond_r_sigma(x, r):
    """
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

    def get_moment(y, moment):
        """
        Returns the (non centered) moment of the distribution y:
        E[y**moment] = \sum_i[index(y_i)^moment * y_i] / \sum_i[y_i]

        :param y: the discrete distribution from which one wants to calculate the moment
        :type y: pandas.Series or np.array
        :param moment: the moment one wants to calcalate (choose 1,2,3, ... )
        :type moment: int
        :return: the moment requested
        :return type: float
        """
        return y.dot(np.arange(len(y)) ** moment) / y.sum()

    def get_centroid(y):
        """
        :param y: the discrete distribution from which one wants to calculate the centroid
        :type y: pandas.Series or np.array
        :return: the centroid of distribution y (aka distribution mean, first moment)
        :return type: float
        """
        return get_moment(y, 1)

    def get_variance(y):
        """
        :param y: the discrete distribution from which one wants to calculate the variance
        :type y: pandas.Series or np.array
        :return: the variance of distribution y
        :return type: float
        """
        return get_moment(y, 2) - get_centroid(y) ** 2

    def get_skew(y):
        """
        Calculates the skew as the third standardized moment.
        Ref: https://en.wikipedia.org/wiki/Skewness#Definition

        :param y: the discrete distribution from which one wants to calculate the skew
        :type y: pandas.Series or np.array
        :return: the skew of distribution y
        :return type: float
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, skew should be 0 and variance 0.  However, in the discrete limit,
        # the skew blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                           get_moment(y, 3) - 3 * get_centroid(y) * variance - get_centroid(y) ** 3
                   ) / get_variance(y) ** (1.5)

    def get_kurtosis(y):
        """
        :param y: the discrete distribution from which one wants to calculate the kurtosis
        :type y: pandas.Series or np.array
        :return: the kurtosis of distribution y
        :return type: float
        """

        variance = get_variance(y)
        # In the limit of a dirac delta, kurtosis should be 3 and variance 0.  However, in the discrete limit,
        # the kurtosis blows up as variance --> 0, hence return nan when variance is smaller than a resolution of 0.5:
        if variance < 0.5:
            return np.nan
        else:
            return (
                           get_moment(y, 4) - 4 * get_centroid(y) * get_moment(y, 3)
                           + 6 * get_moment(y, 2) * get_centroid(y) ** 2 - 3 * get_centroid(y)
                   ) / get_variance(y) ** 2

    calculation = dict(
        centroid=get_centroid,
        variance=get_variance,
        skew=get_skew,
        kurtosis=get_kurtosis
    )

    fft_abs = np.abs(np.fft.rfft(x))

    res = [calculation[config["aggtype"]](fft_abs) for config in param]
    return res
