#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from __future__ import absolute_import, division
import warnings
import pandas as pd
import numpy as np
from builtins import range
from scipy.signal import  find_peaks_cwt, ricker
from scipy.stats import linregress
from time_series_detector.common.tsd_common import DEFAULT_WINDOW


# todo: make sure '_' works in parameter names in all cases, add a warning if not


__all__ = ["time_series_moving_average",
           "time_series_weighted_moving_average",
           "time_series_exponential_weighted_moving_average",
           "time_series_double_exponential_weighted_moving_average",
           "time_series_periodic_features", #####append 的问题？？
           "binned_entropy",
           "quantile",
           "change_quantiles",
           "number_crossing_m",
           "energy_ratio_by_chunks",
           "agg_linear_trend",
           "number_cwt_peaks"]




def set_property(key, value):
    """
    This method returns a decorator that sets the property key of the function to value
    """
    def decorate_func(func):
        setattr(func, key, value)
        if func.__doc__ and key == "fctype":
            func.__doc__ = func.__doc__ + "\n\n    *This function is of type: " + value + "*\n"
        return func
    return decorate_func


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

#

@set_property("fctype", "simple")
def time_series_moving_average(x): ########为什么从后开始计算平均值？？？？
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """
    x = x[4]
    temp = np.mean(x)
    return temp - x[-1]

@set_property("fctype", "simple")
def time_series_weighted_moving_average(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """
    x = x[4]
    w = len(x)
    coefficient = np.array(range(1, w + 1))
    temp__ = ((np.dot(coefficient, x[-w:])) / float(w * (w + 1) / 2)) - x[-1]

    return temp__

@set_property("fctype", "simple")
def time_series_exponential_weighted_moving_average(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """
    x = x[4]
    for j in range(1, 10):
        alpha = j / 10.0
        s = [x[0]]
        for i in range(1, len(x)):
            temp = alpha * x[i] + (1 - alpha) * s[-1]
            s.append(temp)
    return s[-1] - x[-1]

#
@set_property("fctype", "simple")
def time_series_double_exponential_weighted_moving_average(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """
    # list = []
    x = x[4]
    for j1 in range(1, 10, 2):
        for j2 in range(1, 10, 2):
            alpha = j1 / 10.0
            gamma = j2 / 10.0
            s = [x[0]]
            b = [(x[3] - x[0]) / 3]  # s is the smoothing part, b is the trend part ######？？？？？？(x[2]-x[0])/3???
            for i in range(1, len(x)):
                temp1 = alpha * x[i] + (1 - alpha) * (s[-1] + b[-1]) ####?????加(s[i-1] + b[i-1])
                s.append(temp1)
                temp2 = gamma * (s[-1] - s[-2]) + (1 - gamma) * b[-1]  ####?????gamma * (s[i-1] - s[i-2]) + (1 - gamma) * b[i-1]
                b.append(temp2)

    return s[-1] - x[-1]

@set_property("fctype", "simple")
def time_series_periodic_features(x_list):
    """
    :param data_c_left: the time series of historical reference data
    :type data_c_left: pandas.Series
    :param data_c_right: the time series of historical reference data
    :type data_c_right: pandas.Series
    :param data_b_left: the time series of historical reference data
    :type data_b_left: pandas.Series
    :param data_b_right: the time series of historical reference data
    :type data_b_right: pandas.Series
    :param data_a: the time series to calculate the feature of
    :type data_a: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """
    data_c_left, data_c_right, data_b_left, data_b_right, data_a = x_list[0], x_list[1], x_list[2], x_list[3], x_list[4]

    periodic_features = []
    DEFAULT_WINDOW = 14
    # '''
    # Add the absolute value of difference between today and a week ago and its sgn as two features
    # Add the absolute value of difference between today and yesterday and its sgn as two features
    # '''
    temp_value = data_c_left[-1] - data_a[-1]
    periodic_features.append(abs(temp_value))
    if temp_value < 0:
        periodic_features.append(-1)
    else:
        periodic_features.append(1)

    temp_value = data_b_left[-1] - data_a[-1]
    periodic_features.append(abs(temp_value))
    if temp_value < 0:
        periodic_features.append(-1)
    else:
        periodic_features.append(1)

    '''
    If the last value of today is larger than the whole subsequence of a week ago,
    then return the difference between the maximum of the whole subsequence of a week ago and the last value of today.
    Others are similar.
    '''

    periodic_features.append(min(max(data_c_left) - data_a[-1], 0))
    periodic_features.append(min(max(data_c_right) - data_a[-1], 0))
    periodic_features.append(min(max(data_b_left) - data_a[-1], 0))
    periodic_features.append(min(max(data_b_right) - data_a[-1], 0))
    periodic_features.append(max(min(data_c_left) - data_a[-1], 0))
    periodic_features.append(max(min(data_c_right) - data_a[-1], 0))
    periodic_features.append(max(min(data_b_left) - data_a[-1], 0))
    periodic_features.append(max(min(data_b_right) - data_a[-1], 0))
    # periodic_features = np.array(periodic_features)

    return periodic_features

# #
@set_property("fctype", "simple")
def binned_entropy(x, max_bins):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param max_bins: the maximal number of bins
    :type max_bins: int
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    hist, bin_edges = np.histogram(x, bins=max_bins)
    probs = hist / x.size
    return - np.sum(p * np.math.log(p) for p in probs if p != 0)
#
@set_property("fctype", "simple")
def quantile(x, q):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param q: the quantile to calculate
    :type q: float
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(x)
    return pd.Series.quantile(x, q)

@set_property("fctype", "simple")
def change_quantiles(x, ql, qh, isabs, f_agg):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param ql: the lower quantile of the corridor
    :type ql: float
    :param qh: the higher quantile of the corridor
    :type qh: float
    :param isabs: should the absolute differences be taken?
    :type isabs: bool
    :param f_agg: the aggregator function that is applied to the differences in the bin
    :type f_agg: str, name of a numpy function (e.g. mean, var, std, median)

    :return: the value of this feature
    :return type: float
    """
    if ql >= qh:
        ValueError("ql={} should be lower than qh={}".format(ql, qh))

    div = np.diff(x)
    if isabs:
        div = np.abs(div)
    # All values that originate from the corridor between the quantiles ql and qh will have the category 0,
    # other will be np.NaN
    try:
        bin_cat = pd.qcut(x, [ql, qh], labels=False)
        bin_cat_0 = bin_cat == 0
    except ValueError:  # Occurs when ql are qh effectively equal, e.g. x is not long enough or is too categorical
        return 0
    # We only count changes that start and end inside the corridor
    ind = (bin_cat_0 & _roll(bin_cat_0, 1))[1:]
    if sum(ind) == 0:
        return 0
    else:
        ind_inside_corridor = np.where(ind == 1)
        aggregator = getattr(np, f_agg)
        return aggregator(div[ind_inside_corridor])

@set_property("fctype", "simple")
def number_crossing_m(x, m):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param m: the threshold for the crossing
    :type m: float
    :return: the value of this feature
    :return type: int
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    # From https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    positive = x > m
    return np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0].size

@set_property("fctype", "combiner")
def energy_ratio_by_chunks(x, param):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"num_segments": N, "segment_focus": i} with N, i both ints
    :return: the feature values
    :return type: list of tuples (index, data)
    """
    res_data = []
    res_index = []
    x= pd.DataFrame(x)
    full_series_energy = np.sum(x.values ** 2)

    for parameter_combination in param:
        num_segments = parameter_combination["num_segments"]
        segment_focus = parameter_combination["segment_focus"]
        assert segment_focus < num_segments
        assert num_segments > 0

        res_data.append(np.sum(np.array_split(x, num_segments)[segment_focus] ** 2.0)/full_series_energy)
        res_index.append("num_segments_{}__segment_focus_{}".format(num_segments, segment_focus))

    res_data = np.array(res_data)
    return res_data # Materialize as list for Python 3 compatibility with name handling


def _aggregate_on_chunks(x, f_agg, chunk_len):
    """
    :param x: the time series to calculate the aggregation of
    :type x: numpy.ndarray
    :param f_agg: The name of the aggregation function that should be an attribute of the pandas.Series
    :type f_agg: str
    :param chunk_len: The size of the chunks where to aggregate the time series
    :type chunk_len: int
    :return: A list of the aggregation function over the chunks
    :return type: list
    """
    return [getattr(x[i * chunk_len: (i + 1) * chunk_len], f_agg)() for i in range(int(np.ceil(len(x) / chunk_len)))]

@set_property("fctype", "combiner")
def agg_linear_trend(x, param):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    # todo: we could use the index of the DataFrame here

    calculated_agg = {}
    res_data = []
    res_index = []
    x = np.array(x)
    for parameter_combination in param:

        chunk_len = parameter_combination["chunk_len"]
        f_agg = parameter_combination["f_agg"]

        aggregate_result = _aggregate_on_chunks(x, f_agg, chunk_len)
        a = aggregate_result
        a1 = pd.DataFrame(a)
        if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
            if chunk_len >= len(x):
                calculated_agg[f_agg] = {chunk_len: np.NaN}

            else:
                lin_reg_result = linregress(range(len(aggregate_result)), aggregate_result)
                calculated_agg[f_agg] = {chunk_len: lin_reg_result}

        attr = parameter_combination["attr"]

        if chunk_len >= len(x):
            res_data.append(np.NaN)
        else:
            res_data.append(getattr(calculated_agg[f_agg][chunk_len], attr))

        res_index.append("f_agg_\"{}\"__chunk_len_{}__attr_\"{}\"".format(f_agg, chunk_len, attr))
    return res_data


@set_property("fctype", "simple")
def number_cwt_peaks(x, n):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param n: maximum width to consider
    :type n: int
    :return: the value of this feature
    :return type: int
    """
    return len(find_peaks_cwt(vector=x, widths=np.array(list(range(1, n + 1))), wavelet=ricker))