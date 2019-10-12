#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from __future__ import absolute_import, division

import numpy as np
import pandas as pd
import warnings
from builtins import range
from time_series_detector.common.tsd_common import DEFAULT_WINDOW, split_time_series


# todo: make sure '_' works in parameter names in all cases, add a warning if not




__all__ = ["time_series_moving_average",
           "time_series_weighted_moving_average",
           "time_series_exponential_weighted_moving_average",
           "time_series_double_exponential_weighted_moving_average","time_series_periodic_features",
           "binned_entropy",
           "quantile",
           "binned_entropy",
           "change_quantiles",
           "number_crossing_m",
           "energy_ratio_by_chunks",
           "energy_ratio_by_chunks"]





def _roll(a, shift):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])


def time_series_moving_average_get_dict(x):
    def _f():
        for w in range(1, min(50, DEFAULT_WINDOW), 5):
            temp = np.mean(x[-w:])
            temp__ = temp - x[-1]
            name = ("statistical_time_series_moving_average_{}".format(w))
            yield {'{}'.format(name):temp__}
    return list(_f())

def time_series_moving_average(x): ########为什么从后开始计算平均值？？？？
    """
    Returns the difference between the last element of x and the smoothed value after Moving Average Algorithm
    The Moving Average Algorithm is M_{n} = (x_{n-w+1}+...+x_{n})/w, where w is a parameter
    The parameter w is chosen in {1, 6, 11, 16, 21, 26, 31, 36, 41, 46} and the set of parameters can be changed.

    WIKIPEDIA: https://en.wikipedia.org/wiki/Moving_average

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """

    a = time_series_moving_average_get_dict(x)
    return a

def time_series_weighted_moving_average_get_dict(x):
    def _f():
        for w in range(1, min(50, DEFAULT_WINDOW), 5):
            w = min(len(x), w)
            coefficient = np.array(range(1, w + 1))
            temp__ = ((np.dot(coefficient, x[-w:])) / float(w * (w + 1) / 2)) - x[-1]
            name = ("statistical_time_series_weighted_moving_average_{}".format(w))
            yield {'{}'.format(name):temp__}
    return list(_f())

def time_series_weighted_moving_average(x):
    """
    Returns the difference between the last element of x and the smoothed value after Weighted Moving Average Algorithm
    The Moving Average Algorithm is M_{n} = (1*x_{n-w+1}+...+(w-1)*x_{n-1}+w*x_{n})/w, where w is a parameter
    The parameter w is chosen in {1, 6, 11, 16, 21, 26, 31, 36, 41, 46} and the set of parameters can be changed.

    WIKIPEDIA: https://en.wikipedia.org/wiki/Moving_average

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """

    a = time_series_weighted_moving_average_get_dict(x)
    return a

def time_series_exponential_weighted_moving_average_get_dict(x):
    def _f():
        for j in range(1, 10):
            alpha = j / 10.0
            s = [x[0]]
            for i in range(1, len(x)):
                temp = alpha * x[i] + (1 - alpha) * s[-1]
                s.append(temp)
                temp__ = s[-1] - x[-1]
                name = ("statistical_time_series_exponential_weighted_moving_average_j{}_i{}".format(j,i))
                yield {'{}'.format(name):temp__}
    return list(_f())



def time_series_exponential_weighted_moving_average(x):
    """
    Returns the difference between the last element of x and the smoothed value after Exponential Moving Average Algorithm
    The Moving Average Algorithm is s[i] = alpha * x[i] + (1 - alpha) * s[i-1], where alpha is a parameter
    The parameter w is chosen in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} and the set of parameters can be changed.

    WIKIPEDIA: https://en.wikipedia.org/wiki/Exponential_smoothing

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """

    a = time_series_exponential_weighted_moving_average_get_dict(x)
    return a




def time_series_double_exponential_weighted_moving_average_get_dict(x):
    def _f():
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
                    temp__ = s[-1] - x[-1]
                    name = ("statistical_time_series_double_exponential_weighted_moving_average_j1{}_j2{}_i{}".format(j1,j2,i))
                    yield {'{}'.format(name):temp__}
    return list(_f())



def time_series_double_exponential_weighted_moving_average(x):
    """
    Returns the difference between the last element of x and the smoothed value after Double Exponential Moving Average Algorithm
    The Moving Average Algorithm is s[i] = alpha * x[i] + (1 - alpha) * (s[i-1] + b[i-1]), b[i] = gamma * (s[i] - s[i-1]) + (1 - gamma) * b[i-1]
    where alpha and gamma are parameters.
    The parameter alpha is chosen in {0.1, 0.3, 0.5, 0.7, 0.9} and the set of parameters can be changed.
    The parameter gamma is chosen in {0.1, 0.3, 0.5, 0.7, 0.9} and the set of parameters can be changed.

    WIKIPEDIA: https://en.wikipedia.org/wiki/Exponential_smoothing

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """


    a = time_series_double_exponential_weighted_moving_average_get_dict(x)
    return a







def time_series_periodic_features(data_c_left, data_c_right, data_b_left, data_b_right, data_a):
    """
    Returns the difference between the last element of data_a and the last element of data_b_left,
    the difference between the last element of data_a and the last element of data_c_left.

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
    periodic_features = []
    DEFAULT_WINDOW = 14

    '''
    Add the absolute value of difference between today and a week ago and its sgn as two features
    Add the absolute value of difference between today and yesterday and its sgn as two features
    '''

    temp_value = data_c_left[-1] - data_a[-1]
    periodic_features.append({"abs_between_today_and_a_week_ago":abs(temp_value)})
    if temp_value < 0:
        a = -1
        # periodic_features.append(-1)
    else:
        a =1
        # periodic_features.append(1)
    periodic_features.append({"absolute_value_of_today_and_a_week_ago":a})

    temp_value = data_b_left[-1] - data_a[-1]
    periodic_features.append({"abs_between_today_and_yesterday":abs(temp_value)})
    if temp_value < 0:
        a = -1
    else:
        a = 1
    periodic_features.append({"absolute_value_of_today_and_yesterday":a})

    '''
    If the last value of today is larger than the whole subsequence of a week ago,
    then return the difference between the maximum of the whole subsequence of a week ago and the last value of today.
    Others are similar.
    '''

    periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_a_week_ago_add_winlen_before_and_the_last_value_of_today":min(max(data_c_left) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_a_week_ago_add_winlen_after_and_the_last_value_of_today":min(max(data_c_right) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_yesterday_add_winlen_before_and_the_last_value_of_today":min(max(data_b_left) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_yesterday_add_winlen_after_and_the_last_value_of_today":min(max(data_b_right) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_minimum_of_the_whole_subsequence_of_a_week_ago_add_winlen_before_and_the_last_value_of_today":max(min(data_c_left) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_minimum_of_the_whole_subsequence_of_a_week_ago_add_winlen_after_and_the_last_value_of_today":max(min(data_c_right) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_minimum_of_the_whole_subsequence_of_yesterday_add_winlen_before_and_the_last_value_of_today":max(min(data_b_left) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_minimum_of_the_whole_subsequence_of_a_week_ago_add_winlen_after_and_the_last_value_of_today":max(min(data_b_right) - data_a[-1], 0)})

    '''
    If the last value of today is larger than the subsequence of a week ago,
    then return the difference between the maximum of the whole subsequence of a week ago and the last value of today.
    Others are similar.
    '''
    #
    for w in range(1, DEFAULT_WINDOW, DEFAULT_WINDOW / 6):
        periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_a_week_ago_add_winlen_before_oflastnumber_{}_and_the_last_value_of_today{}".format(w,w):min(max(data_c_left[-w:]) - data_a[-1], 0)})
        periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_a_week_ago_add_winlen_after_oflastnumber_{}_and_the_last_value_of_today{}".format(w,w,):min(max(data_c_right[:w]) - data_a[-1], 0)})
        periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_yesterday_add_winlen_before_oflastnumber_{}_and_the_last_value_of_today{}".format(w,w,):min(max(data_b_left[-w:]) - data_a[-1], 0)})
        periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_yesterday_add_winlen_after_oflastnumber_{}_and_the_last_value_of_today{}".format(w,w,):min(max(data_b_right[:w]) - data_a[-1], 0)})
        periodic_features.append({"the_difference_between_the_minimun_of_the_whole_subsequence_of_a_week_ago_add_winlen_before_oflastnumber_{}_and_the_last_value_of_today{}".format(w,w,):max(min(data_c_left[-w:]) - data_a[-1], 0)})
        periodic_features.append({"the_difference_between_the_minimum_of_the_whole_subsequence_of_a_week_ago_add_winlen_after_oflastnumber_{}_and_the_last_value_of_today{}".format(w,w,):max(min(data_c_right[:w]) - data_a[-1], 0)})
        periodic_features.append({"the_difference_between_the_miniimum_of_the_whole_subsequence_of_yesterday_add_winlen_before_oflastnumber_{}_and_the_last_value_of_today{}".format(w,w,):max(min(data_b_left[-w:]) - data_a[-1], 0)})
        periodic_features.append({"the_difference_between_the_minimum_of_the_whole_subsequence_of_yesterday_add_winlen_after_oflastnumber_{}_and_the_last_value_of_today{}".format(w,w,):max(min(data_b_right[:w]) - data_a[-1], 0)})

    '''
    Add the difference of mean values between two subsequences
    '''

    for w in range(1, DEFAULT_WINDOW, DEFAULT_WINDOW / 9):
        temp_value = np.mean(data_c_left[-w:]) - np.mean(data_a[-w:])
        periodic_features.append({"abs_of_mean_values_between_data_c_left_and_data_a_lenth_{}".format(w):abs(temp_value)})
        if temp_value < 0:
            a = -1
            # periodic_features.append(-1)
        else:
            a = 1
        periodic_features.append({"data_c_left_mean_larger_than_data_a_mean_lenth_{}".format(w):a})

        temp_value = np.mean(data_c_right[:w]) - np.mean(data_a[-w:])
        periodic_features.append({"abs_of_mean_values_between_data_c_right_and_data_a_lenth_{}".format(w):abs(temp_value)})
        if temp_value < 0:
            a = -1
        else:
            a = 1
        periodic_features.append({"data_c_right_mean_larger_than_data_a_mean_lenth_{}".format(w):a})

        temp_value = np.mean(data_b_left[-w:]) - np.mean(data_a[-w:])
        periodic_features.append({"abs_of_mean_values_between_data_b_left_and_data_a_lenth_{}".format(w):abs(temp_value)})
        if temp_value < 0:
            a = -1
        else:
            a = 1
        periodic_features.append({"data_b_left_mean_larger_than_data_a_mean_lenth_{}".format(w):a})

        temp_value = np.mean(data_b_right[:w]) - np.mean(data_a[-w:])
        periodic_features.append({"abs_of_mean_values_between_data_b_right_and_data_a_lenth_{}".format(w):abs(temp_value)})
        if temp_value < 0:
            a = -1
        else:
            a = 1
        periodic_features.append({"data_b_right_mean_larger_than_data_a_mean_lenth_{}".format(w):a})

    step = DEFAULT_WINDOW / 6

    for w in range(1, DEFAULT_WINDOW, DEFAULT_WINDOW/6):
        periodic_features.append({"the_difference_between_the_maximum_of_the_subsequence_of_data_a_from_{}_the_last_value_of_today_{}".format(w,w):min(max(data_a[w - 1:w + step]) - data_a[-1], 0)})
        periodic_features.append({"the_difference_between_the_minimum_of_the_subsequence_of_data_a_from_{}_the_last_value_of_today_{}".format(w,w):max(min(data_a[w - 1:w + step]) - data_a[-1], 0)})


    return periodic_features

# add yourself fitting features here...

def binned_entropy(x, max_bins):
    """
    First bins the values of x into max_bins equidistant bins.
    Then calculates the value of

    .. math::

        - \\sum_{k=0}^{min(max\\_bins, len(x))} p_k log(p_k) \\cdot \\mathbf{1}_{(p_k > 0)}

    where :math:`p_k` is the percentage of samples in bin :math:`k`.

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

@set_property("fctype", "simple")
def quantile(x, q):
    """
    Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param q: the quantile to calculate
    :type q: float
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(x)
    return pd.Series.quantile(x, q)


def change_quantiles(x, ql, qh, isabs, f_agg):
    """
    First fixes a corridor given by the quantiles ql and qh of the distribution of x.
    Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

    Think about selecting a corridor on the
    y-Axis and only calculating the mean of the absolute change of the time series inside this corridor.

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

def change_quantiles(x, ql, qh, isabs, f_agg):
    """
    First fixes a corridor given by the quantiles ql and qh of the distribution of x.
    Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

    Think about selecting a corridor on the
    y-Axis and only calculating the mean of the absolute change of the time series inside this corridor.

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

def number_crossing_m(x, m):
    """
    Calculates the number of crossings of x on m. A crossing is defined as two sequential values where the first value
    is lower than m and the next is greater, or vice-versa. If you set m to zero, you will get the number of zero
    crossings.

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


def energy_ratio_by_chunks(x, param):
    """
    Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole
    series.

    Takes as input parameters the number num_segments of segments to divide the series into and segment_focus
    which is the segment number (starting at zero) to return a feature on.

    If the length of the time series is not a multiple of the number of segments, the remaining data points are
    distributed on the bins starting from the first. For example, if your time series consists of 8 entries, the
    first two bins will contain 3 and the last two values, e.g. `[ 0.,  1.,  2.], [ 3.,  4.,  5.]` and `[ 6.,  7.]`.

    Note that the answer for `num_segments = 1` is a trivial "1" but we handle this scenario
    in case somebody calls it. Sum of the ratios should be 1.0.

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
    # return list(zip(res_index, res_data)) # Materialize as list for Python 3 compatibility with name handling
    return res_data # Materialize as list for Python 3 compatibility with name handling


def energy_ratio_by_chunks(x, param):
    """
    Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole
    series.

    Takes as input parameters the number num_segments of segments to divide the series into and segment_focus
    which is the segment number (starting at zero) to return a feature on.

    If the length of the time series is not a multiple of the number of segments, the remaining data points are
    distributed on the bins starting from the first. For example, if your time series consists of 8 entries, the
    first two bins will contain 3 and the last two values, e.g. `[ 0.,  1.,  2.], [ 3.,  4.,  5.]` and `[ 6.,  7.]`.

    Note that the answer for `num_segments = 1` is a trivial "1" but we handle this scenario
    in case somebody calls it. Sum of the ratios should be 1.0.

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
    # return list(zip(res_index, res_data)) # Materialize as list for Python 3 compatibility with name handling
    return res_data # Materialize as list for Python 3 compatibility with name handling
