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

import tsfresh.feature_extraction.feature_calculators as ts_feature_calculators
# import tsfresh.feature_extraction.feature_calculators as ts_feature_calculators_without_param
import time_series_detector.feature.feature_calculators_withparam as ts_feature_calculators_without_param
import warnings
import pandas as pd
from feature.setting import ComprehensiveFCParameters


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
###########################################新加入的tsfresh内容##########################################
def length(x):
    """
    Returns the length of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: int
    """
    return len(x)





###
def _do_extraction_on_chunk(x, default_fc_parameters = None):
    """
    Main function of this module: use the feature calculators defined in the
    default_fc_parameters or kind_to_fc_parameters parameters and extract all
    features on the chunk.

    The chunk consists of the chunk id, the chunk kind and the data (as a Series),
    which is then converted to a numpy array - so a single time series.

    Returned is a list of the extracted features. Each one is a dictionary consisting of
    { "variable": the feature name in the format <kind>__<feature>__<parameters>,
      "value": the number value of the feature,
      "id": the id of the chunk }

    The <parameters> are in the form described in :mod:`~tsfresh.utilities.string_manipulation`.

    :param chunk: A tuple of sample_id, kind, data
    :param default_fc_parameters: A dictionary of feature calculators.
    :param kind_to_fc_parameters: A dictionary of fc_parameters for special kinds or None.
    :return: A list of calculated features.
    """
    data = x
    fc_paxrameters = ComprehensiveFCParameters()

    def _f():
        for function_name, parameter_list in fc_paxrameters.items():
            func = getattr(ts_feature_calculators_without_param, function_name)

            # If the function uses the index, pass is at as a pandas Series.
            # Otherwise, convert to numpy array
            if getattr(func, 'input', False) == 'pd.Series':
                # If it has a required index type, check that the data has the right index type.
                index_type = getattr(func, 'index_type', None)

                if index_type is not None:
                    try:
                        assert isinstance(data.index, index_type)
                    except AssertionError:
                        warnings.warn(
                            "{} requires the data to have a index of type {}. Results will "
                            "not be calculated".format(function_name, index_type)
                        )
                        continue
                a = data
            else:
                # a = pd.DataFrame(data)
                a = data
                # a = pd.DataFrame(data)
                # b = a.values
                # b =a
            if func.fctype == "combiner":
                result = func(a, param=parameter_list)
            else:
                if parameter_list:
                    result = (func(a, **param) for param in
                              parameter_list)
                else:
                    result = [("", func(a))]


            for key, item in enumerate(result):
                feature_name = "__" + func.__name__
                if key:
                    feature_name += "__" + str(key)
                # yield {"variable": feature_name, "value": item} ##origin
                yield {"{}".format(feature_name):item}

                # yield {feature_name:item}

    return list(_f())
    # return k_format_changed

##############################################################################################################################
##############################################################################################################################



###########################################新加入的tsfresh内容##########################################


#
def get_statistical_features(x):
    statistical_features = [
        {"time_series_maximum_statistical_features":time_series_maximum(x)},
        {"time_series_minimum_statistical_features":time_series_minimum(x)},
        {"time_series_mean_statistical_features":time_series_mean(x)},
        {"time_series_variance_statistical_features":time_series_variance(x)},
        {"time_series_standard_deviation_statistical_features":time_series_standard_deviation(x)},
        {"time_series_skewness_statistical_features":time_series_skewness(x)},
        {"time_series_kurtosis_statistical_features":time_series_kurtosis(x)},
        {"time_series_median_statistical_features":time_series_median(x)},
        {"time_series_sum_values_statistical_features":time_series_sum_values(x)},
        {"time_series_range_statistical_features":time_series_range(x)},
        {"time_series_abs_energy_statistical_features":time_series_abs_energy(x)},
        {"time_series_absolute_sum_of_changes_statistical_features":time_series_absolute_sum_of_changes(x)},
        {"time_series_variance_larger_than_std_statistical_features":time_series_variance_larger_than_std(x)},
        {"time_series_count_above_mean_statistical_features":time_series_count_above_mean(x)},
        {"time_series_count_below_mean_statistical_features":time_series_count_below_mean(x)},
        {"time_series_first_location_of_maximum_statistical_features":time_series_first_location_of_maximum(x)},
        {"time_series_first_location_of_minimum_statistical_features":time_series_first_location_of_minimum(x)},
        {"time_series_last_location_of_maximum_statistical_features":time_series_last_location_of_maximum(x)},
        {"time_series_last_location_of_minimum_statistical_features":time_series_last_location_of_minimum(x)},
        {"int(time_series_has_duplicate(x))_statistical_features":int(time_series_has_duplicate(x))},
        {"int(time_series_has_duplicate_min(x)_statistical_features)":int(time_series_has_duplicate_min(x))},
        {"time_series_longest_strike_above_mean_statistical_features":time_series_longest_strike_above_mean(x)},
        {"time_series_longest_strike_below_mean_statistical_features":time_series_longest_strike_below_mean(x)},
        {"time_series_mean_abs_change_statistical_features":time_series_mean_abs_change(x)},
        {"time_series_mean_change_statistical_features":time_series_mean_change(x)},
        {"time_series_percentage_of_reoccurring_datapoints_to_all_datapoints_statistical_features":time_series_percentage_of_reoccurring_datapoints_to_all_datapoints(x)},
        {"time_series_ratio_value_number_to_time_series_length_statistical_features":time_series_ratio_value_number_to_time_series_length(x)},
        {"time_series_sum_of_reoccurring_data_points_statistical_features":time_series_sum_of_reoccurring_data_points(x)},
        {"time_series_sum_of_reoccurring_values_statistical_features":time_series_sum_of_reoccurring_values(x)},
        {"length":length(x)}]
    # append yourself statistical features here...
    return statistical_features




#######################################################################################################################



def get_parameters_features(x, default_fc_parameters=None):
    if default_fc_parameters is None:
        default_fc_parameters = ComprehensiveFCParameters()
    k = _do_extraction_on_chunk(x)
    return k




