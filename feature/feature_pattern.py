#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Tencent is pleased to support the open source community by making Metis available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import numpy as np
import tsfresh.feature_extraction.feature_calculators as ts_feature_calculators
from time_series_detector.common.tsd_common import DEFAULT_WINDOW, split_time_series
from builtins import range
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.signal import cwt, ricker, welch
from scipy.stats import linregress
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import acf, pacf


__all__ = [	"time_series_autocorrelation",
               "time_series_coefficient_of_variation",
               "time_series_value_distribution",
               "time_series_daily_parts_value_distribution",
               "time_series_daily_parts_value_distribution_with_threshold",
               "time_series_binned_entropy",
               "approximate_entropy",
               "sample_entropy",
               "cwt_coefficients",
               "fft_coefficient",
               "ar_coefficient",
               "cid_ce",
               "partial_autocorrelation",
               "agg_autocorrelation",
               "symmetry_looking",
               "time_reversal_asymmetry_statistic",
               "c3",
               "spkt_welch_density",
               "linear_trend",
                "linear_trend_timewise"

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


def time_series_autocorrelation(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param lag: the lag
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    lag = int((len(x) - 3) / 5)
    if np.sqrt(np.var(x)) < 1e-10:
        return 0
    return ts_feature_calculators.autocorrelation(x, lag)


def time_series_coefficient_of_variation(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    if np.sqrt(np.var(x)) < 1e-10:
        return 0
    return np.mean(x) / np.sqrt(np.var(x))



def time_series_binned_entropy(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param max_bins: the maximal number of bins
    :type max_bins: int
    :return: the value of this feature
    :return type: float
    """
    list = []
    max_bins = [2, 4, 6, 8, 10, 20]
    for value in max_bins:
        temp__ = ts_feature_calculators.binned_entropy(x, value)
        name = ("time_series_binned_entropy_max_bins_{}".format(value))
        list.append({'{}'.format(name): temp__})
    return list




def time_series_value_distribution(x):
    """
    :param x: normalized time series
    :type x: pandas.Series
    :return: the values of this feature
    :return type: list
    """
    thresholds = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0, 1.0]
    a = list(np.histogram(x, bins=thresholds)[0] / float(len(x)))
    return list(np.histogram(x, bins=thresholds)[0] / float(len(x)))




def time_series_daily_parts_value_distribution(x):
    """
    :param x: normalized time series
    :type x: pandas.Series
    :return: the values of this feature
    :return type: list
    """
    thresholds = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0, 1.0]
    split_value_list = split_time_series(x, DEFAULT_WINDOW)
    data_c = split_value_list[0] + split_value_list[1][1:]
    data_b = split_value_list[2] + split_value_list[3][1:]
    data_a = split_value_list[4]
    count_c = list(np.histogram(data_c, bins=thresholds)[0])
    count_b = list(np.histogram(data_b, bins=thresholds)[0])
    count_a = list(np.histogram(data_a, bins=thresholds)[0])
    return list(np.array(count_c) / float(len(data_c))) + list(np.array(count_b) / float(len(data_b))) + list(np.array(count_a) / float(len(data_a)))


def time_series_daily_parts_value_distribution_with_threshold(x):
    """
    :param x: normalized time series
    :type x: pandas.Series
    :return: 6 values of this feature
    :return type: list
    """
    threshold = 0.01
    split_value_list = split_time_series(x, DEFAULT_WINDOW)
    data_c = split_value_list[0] + split_value_list[1][1:]
    data_b = split_value_list[2] + split_value_list[3][1:]
    data_a = split_value_list[4]

    # the number of elements in time series which is less than threshold:
    nparray_data_c_threshold = np.array(data_c)
    nparray_data_c_threshold[nparray_data_c_threshold < threshold] = -1
    nparray_data_b_threshold = np.array(data_b)
    nparray_data_b_threshold[nparray_data_b_threshold < threshold] = -1
    nparray_data_a_threshold = np.array(data_a)
    nparray_data_a_threshold[nparray_data_a_threshold < threshold] = -1

    # the total number of elements in time series which is less than threshold:
    nparray_threshold_count = (nparray_data_c_threshold == -1).sum() + (nparray_data_b_threshold == -1).sum() + (nparray_data_a_threshold == -1).sum()

    if nparray_threshold_count == 0:
        features = [{"number_of_elements_less_than_threshould_lastweek":0}, {"number_of_elements_less_than_threshould_yesterday":0}, {"number_of_elements_less_than_threshould_today",0}]
    else:
        features = [
            {"number_of_elements_less_than_threshould_lastweek_take_up_total_threshould_number":(nparray_data_c_threshold == -1).sum() / float(nparray_threshold_count)},
            {"number_of_elements_less_than_threshould_yesterday_take_up_total_threshould_number":(nparray_data_b_threshold == -1).sum() / float(nparray_threshold_count)},
            {"number_of_elements_less_than_threshould_today_take_up_total_threshould_number":(nparray_data_a_threshold == -1).sum() / float(nparray_threshold_count)}
        ]

    features.extend([
        {"number_of_elements_less_than_threshould_lastweek_take_up_total_lastweek_number":(nparray_data_c_threshold == -1).sum() / float(len(data_c))},
        {"number_of_elements_less_than_threshould_lastweek_take_up_total_yesterday_number":(nparray_data_b_threshold == -1).sum() / float(len(data_b))},
        {"number_of_elements_less_than_threshould_lastweek_take_up_total_today_number":(nparray_data_a_threshold == -1).sum() / float(len(data_a))}
    ])
    return features


def time_series_window_parts_value_distribution_with_threshold(x):
    """
    :param x: normalized time series
    :type x: pandas.Series
    :return: 5 values of this feature
    :return type: list
    """
    list = []
    threshold = 0.01
    split_value_list = split_time_series(x, DEFAULT_WINDOW)

    count_list = []
    a = 0
    for value_list in split_value_list:
        nparray_threshold = np.array(value_list)
        nparray_threshold[nparray_threshold < threshold] = -1
        temp = (nparray_threshold == -1).sum()
        count_list.append((nparray_threshold == -1).sum())
        name = ("time_series_window_parts_value_distribution_with_threshold_{}".format(a))
        a = a + 1
        if sum(count_list) == 0:
            features = [{'time_series_window_parts_value_distribution_with_threshold_Ais0': 0},
                        {'time_series_window_parts_value_distribution_with_threshold_bis0': 0},
                        {'time_series_window_parts_value_distribution_with_threshold_cis0': 0},
                        {'time_series_window_parts_value_distribution_with_threshold_Dis0': 0},
                        {'time_series_window_parts_value_distribution_with_threshold_Eis0': 0}]

        else:
            features = temp / float((DEFAULT_WINDOW + 1))
        list.append({'{}'.format(name): features})

    return list


def approximate_entropy(x, m, r):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param m: Length of compared run of data
    :type m: int
    :param r: Filtering level, must be positive
    :type r: float

    :return: Approximate entropy
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)

    N = x.size
    r *= np.std(x)
    if r < 0:
        raise ValueError("Parameter r must be positive.")
    if N <= m+1:
        return 0

    def _phi(m):
        x_re = np.array([x[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]),
                          axis=2) <= r, axis=0) / (N-m+1)
        return np.sum(np.log(C)) / (N - m + 1.0)

    return np.abs(_phi(m) - _phi(m + 1))


def sample_entropy(x):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray

    :return: the value of this feature
    :return type: float
    """
    x = np.array(x)

    sample_length = 1 # number of sequential points of the time series
    tolerance = 0.2 * np.std(x) # 0.2 is a common value for r - why?

    n = len(x)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((1, 1))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((1, 1))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = x[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]

    N = n * (n - 1) / 2
    B = np.vstack(([N], B[0]))

    similarity_ratio = A / B
    se = -1 * np.log(similarity_ratio)
    se = np.reshape(se, -1)
    return se[0]


def cwt_coefficients(x, param):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"widths":x, "coeff": y, "w": z} with x array of int and y,z int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    calculated_cwt = {}
    res = []
    indices = []

    for parameter_combination in param:
        widths = parameter_combination["widths"]
        w = parameter_combination["w"]
        coeff = parameter_combination["coeff"]

        if widths not in calculated_cwt:
            calculated_cwt[widths] = cwt(x, ricker, widths)

        calculated_cwt_for_widths = calculated_cwt[widths]

        indices += ["widths_{}__coeff_{}__w_{}".format(widths, coeff, w)]

        i = widths.index(w)
        if calculated_cwt_for_widths.shape[1] <= coeff:
            res += [np.NaN]
        else:
            res += [calculated_cwt_for_widths[i, coeff]]

    return res

def fft_coefficient(x, param):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"coeff": x, "attr": s} with x int and x >= 0, s str and in ["real", "imag",
        "abs", "angle"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    assert min([config["coeff"] for config in param]) >= 0, "Coefficients must be positive or zero."
    assert set([config["attr"] for config in param]) <= set(["imag", "real", "abs", "angle"]), \
        'Attribute must be "real", "imag", "angle" or "abs"'

    fft = np.fft.rfft(x)

    def complex_agg(x, agg):
        if agg == "real":
            return x.real
        elif agg == "imag":
            return x.imag
        elif agg == "abs":
            return np.abs(x)
        elif agg == "angle":
            return np.angle(x, deg=True)

    return (complex_agg(fft[config["coeff"]], config["attr"]) if config["coeff"] < len(fft)
            else np.NaN for config in param)

def ar_coefficient(x, param):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"coeff": x, "k": y} with x,y int
    :type param: list
    :return x: the different feature values
    :return type: pandas.Series
    """
    calculated_ar_params = {}

    x_as_list = list(x)
    calculated_AR = AR(x_as_list)

    res = {}

    for parameter_combination in param:
        k = parameter_combination["k"]
        p = parameter_combination["coeff"]

        column_name = "k_{}__coeff_{}".format(k, p)

        if k not in calculated_ar_params:
            try:
                calculated_ar_params[k] = calculated_AR.fit(maxlag=k, solver="mle").params
            except (LinAlgError, ValueError):
                calculated_ar_params[k] = [np.NaN]*k

        mod = calculated_ar_params[k]

        if p <= k:
            try:
                res[column_name] = mod[p]
            except IndexError:
                res[column_name] = 0
        else:
            res[column_name] = np.NaN

    return [(value) for key, value in res.items()]


def cid_ce(x, normalize):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param normalize: should the time series be z-transformed?
    :type normalize: bool

    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if normalize:
        s = np.std(x)
        if s!=0:
            x = (x - np.mean(x))/s
        else:
            return 0.0

    x = np.diff(x)
    return np.sqrt(np.dot(x, x))

def partial_autocorrelation(x, param):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"lag": val} with int val indicating the lag to be returned
    :type param: list
    :return: the value of this feature
    :return type: float
    """
    # Check the difference between demanded lags by param and possible lags to calculate (depends on len(x))
    max_demanded_lag = max([lag["lag"] for lag in param])
    n = len(x)

    # Check if list is too short to make calculations
    if n <= 1:
        pacf_coeffs = [np.nan] * (max_demanded_lag + 1)
    else:
        if (n <= max_demanded_lag):
            max_lag = n - 1
        else:
            max_lag = max_demanded_lag
        pacf_coeffs = list(pacf(x, method="ld", nlags=max_lag))
        pacf_coeffs = pacf_coeffs + [np.nan] * max(0, (max_demanded_lag - max_lag))

    # return [("lag_{}".format(lag["lag"]), pacf_coeffs[lag["lag"]]) for lag in param]
    return [(pacf_coeffs[lag["lag"]]) for lag in param]

def agg_autocorrelation(x, param):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"f_agg": x, "maxlag", n} with x str, the name of a numpy function
                  (e.g. "mean", "var", "std", "median"), its the name of the aggregator function that is applied to the
                  autocorrelations. Further, n is an int and the maximal number of lags to consider.
    :type param: list
    :return: the value of this feature
    :return type: float
    """
    # if the time series is longer than the following threshold, we use fft to calculate the acf
    THRESHOLD_TO_USE_FFT = 1250
    var = np.var(x)
    n = len(x)
    max_maxlag = max([config["maxlag"] for config in param])

    if np.abs(var) < 10**-10 or n == 1:
        a = [0] * len(x)
    else:
        a = acf(x, unbiased=True, fft=n > THRESHOLD_TO_USE_FFT, nlags=max_maxlag)[1:]
    return [(getattr(np, config["f_agg"])(a[:int(config["maxlag"])])) for config in param]

def symmetry_looking(x, param):
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
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    return [(mean_median_difference < (r["r"] * max_min_difference))
            for r in param]


def time_reversal_asymmetry_statistic(x, lag):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    n = len(x)
    x = np.asarray(x)
    if 2 * lag >= n:
        return 0
    else:
        one_lag = _roll(x, -lag)
        two_lag = _roll(x, 2 * -lag)
        return np.mean((two_lag * two_lag * one_lag - one_lag * x * x)[0:(n - 2 * lag)])


def c3(x, lag):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    n = x.size
    if 2 * lag >= n:
        return 0
    else:
        return np.mean((_roll(x, 2 * -lag) * _roll(x, -lag) * x)[0:(n - 2 * lag)])

def spkt_welch_density(x, param):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"coeff": x} with x int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    freq, pxx = welch(x, nperseg=min(len(x), 256))
    coeff = [config["coeff"] for config in param]

    if len(pxx) <= np.max(coeff):  # There are fewer data points in the time series than requested coefficients

        # filter coefficients that are not contained in pxx
        reduced_coeff = [coefficient for coefficient in coeff if len(pxx) > coefficient]
        not_calculated_coefficients = [coefficient for coefficient in coeff
                                       if coefficient not in reduced_coeff]

        # Fill up the rest of the requested coefficients with np.NaNs
        return list(pxx[reduced_coeff]) + [np.NaN] * len(not_calculated_coefficients)
    else:
        return list(pxx[coeff])


def linear_trend(x, param):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"attr": x} with x an string, the attribute name of the regression model
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    # todo: we could use the index of the DataFrame here
    linReg = linregress(range(len(x)), x)

    return [getattr(linReg, config["attr"])
            for config in param]



def linear_trend_timewise(x, param):
    """
    :param x: the time series to calculate the feature of. The index must be datetime.
    :type x: pandas.Series
    :param param: contains dictionaries {"attr": x} with x an string, the attribute name of the regression model
    :type param: list
    :return: the different feature values
    :return type: list
    """
    ix = x.index

    # Get differences between each timestamp and the first timestamp in seconds.
    # Then convert to hours and reshape for linear regression
    times_seconds = (ix - ix[0]).total_seconds()
    times_hours = np.asarray(times_seconds / float(3600))

    linReg = linregress(times_hours, x.values)

    return [getattr(linReg, config["attr"])
            for config in param]

