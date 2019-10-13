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

    list = []
    for w in range(1, min(50, DEFAULT_WINDOW), 5):
        temp = np.mean(x[-w:])
        temp__ = temp - x[-1]
        name = ("statistical_time_series_moving_average_{}".format(w))
        list.append({'{}'.format(name):temp__})

    return list


def time_series_weighted_moving_average(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """

    list = []
    for w in range(1, min(50, DEFAULT_WINDOW), 5):
        w = min(len(x), w)
        coefficient = np.array(range(1, w + 1))
        temp__ = ((np.dot(coefficient, x[-w:])) / float(w * (w + 1) / 2)) - x[-1]
        name = ("statistical_time_series_weighted_moving_average_{}".format(w))
        list.append({'{}'.format(name):temp__})
    return list


def time_series_exponential_weighted_moving_average(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """

    list = []
    for j in range(1, 10):
        alpha = j / 10.0
        s = [x[0]]
        for i in range(1, len(x)):
            temp = alpha * x[i] + (1 - alpha) * s[-1]
            s.append(temp)
            temp__ = s[-1] - x[-1]
            name = ("statistical_time_series_exponential_weighted_moving_average_j{}_i{}".format(j,i))
            list.append({'{}'.format(name):temp__})
    return list



def time_series_double_exponential_weighted_moving_average(x):
    """
    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: list with float
    """
    list = []
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
                list.append({'{}'.format(name):temp__})
    return list

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
    periodic_features.append({"abs_between_today_and_a_week_ago":abs(temp_value)})
    if temp_value < 0:
        a = -1
    else:
        a =1
    periodic_features.append({"absolute_value_of_today_and_a_week_ago":a})

    temp_value = data_b_left[-1] - data_a[-1]
    periodic_features.append({"abs_between_today_and_yesterday":abs(temp_value)})
    if temp_value < 0:
        a = -1
    else:
        a = 1
    periodic_features.append({"absolute_value_of_today_and_yesterday":a})

    # '''
    # If the last value of today is larger than the whole subsequence of a week ago,
    # then return the difference between the maximum of the whole subsequence of a week ago and the last value of today.
    # Others are similar.
    # '''

    periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_a_week_ago_add_winlen_before_and_the_last_value_of_today":min(max(data_c_left) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_a_week_ago_add_winlen_after_and_the_last_value_of_today":min(max(data_c_right) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_yesterday_add_winlen_before_and_the_last_value_of_today":min(max(data_b_left) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_maximum_of_the_whole_subsequence_of_yesterday_add_winlen_after_and_the_last_value_of_today":min(max(data_b_right) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_minimum_of_the_whole_subsequence_of_a_week_ago_add_winlen_before_and_the_last_value_of_today":max(min(data_c_left) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_minimum_of_the_whole_subsequence_of_a_week_ago_add_winlen_after_and_the_last_value_of_today":max(min(data_c_right) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_minimum_of_the_whole_subsequence_of_yesterday_add_winlen_before_and_the_last_value_of_today":max(min(data_b_left) - data_a[-1], 0)})
    periodic_features.append({"the_difference_between_the_minimum_of_the_whole_subsequence_of_a_week_ago_add_winlen_after_and_the_last_value_of_today":max(min(data_b_right) - data_a[-1], 0)})

    # '''
    # If the last value of today is larger than the subsequence of a week ago,
    # then return the difference between the maximum of the whole subsequence of a week ago and the last value of today.
    # Others are similar.
    # '''
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

    # '''
    # Add the difference of mean values between two subsequences
    # '''

    for w in range(1, DEFAULT_WINDOW, DEFAULT_WINDOW / 9):
        temp_value = np.mean(data_c_left[-w:]) - np.mean(data_a[-w:])
        periodic_features.append({"abs_of_mean_values_between_data_c_left_and_data_a_lenth_{}".format(w):abs(temp_value)})
        if temp_value < 0:
            a = -1
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






def get_fitting_features(x_list):
    fitting_features = []
    fitting_features.extend(time_series_moving_average(x_list[4]))
    fitting_features.extend(time_series_weighted_moving_average(x_list[4]))
    fitting_features.extend(time_series_exponential_weighted_moving_average(x_list[4]))
    fitting_features.extend(time_series_double_exponential_weighted_moving_average(x_list[4]))
    #
    fitting_features.extend(time_series_periodic_features(x_list[0], x_list[1], x_list[2], x_list[3], x_list[4]))
    # append yourself fitting features here...

    return fitting_features
