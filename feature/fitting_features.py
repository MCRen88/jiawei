#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Tencent is pleased to support the open source community by making Metis available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from time_series_detector.common.tsd_common import *


def time_series_moving_average(x): ########为什么从后开始计算平均值？？？？
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """

    temp = np.mean(x)
    return temp - x[-1]


def time_series_weighted_moving_average(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """

    w = len(x)
    coefficient = np.array(range(1, w + 1))
    temp__ = ((np.dot(coefficient, x[-w:])) / float(w * (w + 1) / 2)) - x[-1]

    return temp__


def time_series_exponential_weighted_moving_average(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """
    for j in range(1, 10):
        alpha = j / 10.0
        s = [x[0]]
        for i in range(1, len(x)):
            temp = alpha * x[i] + (1 - alpha) * s[-1]
            s.append(temp)

    return s[-1] - x[-1]



def time_series_double_exponential_weighted_moving_average(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """
    # list = []
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

def time_series_periodic_features(data_c_left, data_c_right, data_b_left, data_b_right, data_a):
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

    return periodic_features





def get_fitting_features(x_list):
    """
    :param x: normalized splited time series
    :return: list of local anomaly and periodic features
    """
    fitting_features = []
    fitting_features.extend(time_series_moving_average(x_list[4]))
    fitting_features.extend(time_series_weighted_moving_average(x_list[4]))
    fitting_features.extend(time_series_exponential_weighted_moving_average(x_list[4]))
    fitting_features.extend(time_series_double_exponential_weighted_moving_average(x_list[4]))
    #
    fitting_features.extend(time_series_periodic_features(x_list[0], x_list[1], x_list[2], x_list[3], x_list[4]))

    return fitting_features
