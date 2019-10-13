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
import time_series_detector.feature.feature_calculators_withoutparam as ts_feature_calculators_without_param
import warnings
from time_series_detector.feature.setting import ComprehensiveFCParameters


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

def length(x):
    """
    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: int
    """
    return len(x)



def _do_extraction_on_chunk(x, default_fc_parameters = None):
    """
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
                a = data

            if func.fctype == "combiner":
                result = func(a, param=parameter_list)
                # result = a
            #
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
                yield {"{}".format(feature_name):item}

    return list(_f())

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
    return statistical_features


#######################################################################################################################


#
def get_parameters_features(x, default_fc_parameters=None):
    if default_fc_parameters is None:
        default_fc_parameters = ComprehensiveFCParameters()
    k = _do_extraction_on_chunk(x)
    return k

