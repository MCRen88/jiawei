#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Tencent is pleased to support the open source community by making Metis available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""
import warnings
import statistical_features
import fitting_features
import classification_features
# import feature_calculator
from time_series_detector.common import tsd_common
from tsfresh import defaults
from tsfresh.feature_extraction import feature_calculators
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
import pandas as pd
from time_series_detector.common.tsd_common import *



#
# def _do_extraction_on_chunk(chunk, default_fc_parameters = None):
#     """
#     Main function of this module: use the feature calculators defined in the
#     default_fc_parameters or kind_to_fc_parameters parameters and extract all
#     features on the chunk.
#
#     The chunk consists of the chunk id, the chunk kind and the data (as a Series),
#     which is then converted to a numpy array - so a single time series.
#
#     Returned is a list of the extracted features. Each one is a dictionary consisting of
#     { "variable": the feature name in the format <kind>__<feature>__<parameters>,
#       "value": the number value of the feature,
#       "id": the id of the chunk }
#
#     The <parameters> are in the form described in :mod:`~tsfresh.utilities.string_manipulation`.
#
#     :param chunk: A tuple of sample_id, kind, data
#     :param default_fc_parameters: A dictionary of feature calculators.
#     :param kind_to_fc_parameters: A dictionary of fc_parameters for special kinds or None.
#     :return: A list of calculated features.
#     """
#     data = chunk
#     fc_parameters = default_fc_parameters
#
#     def _f():
#         for function_name, parameter_list in fc_parameters.items():
#             func = getattr(feature_calculators, function_name)
#
#             # If the function uses the index, pass is at as a pandas Series.
#             # Otherwise, convert to numpy array
#             if getattr(func, 'input', False) == 'pd.Series':
#                 # If it has a required index type, check that the data has the right index type.
#                 index_type = getattr(func, 'index_type', None)
#                 if index_type is not None:
#                     try:
#                         assert isinstance(data.index, index_type)
#                     except AssertionError:
#                         warnings.warn(
#                             "{} requires the data to have a index of type {}. Results will "
#                             "not be calculated".format(function_name, index_type)
#                         )
#                         continue
#                 x = data
#             else:
#                 x = data.values
#
#             if func.fctype == "combiner":
#                 result = func(x, param=parameter_list)
#             else:
#                 if parameter_list:
#                     # result = ((convert_to_output_format(param), func(x, **param)) for param in
#                     #           parameter_list)
#                     result = (func(x, **param) for param in
#                               parameter_list)
#                 else:
#                     result = [("", func(x))]
#
#             for key, item in enumerate(result):
#                 feature_name = "__" + func.__name__
#                 if key:
#                     feature_name += "__" + str(key)
#                 yield {"variable": feature_name, "value": item}
#
#     return list(_f())

# def get_fitting_features(x_list):
#     fitting_features = []
#     fitting_features.extend(time_series_moving_average(x_list[4]))
#     fitting_features.extend(time_series_weighted_moving_average(x_list[4]))
#     fitting_features.extend(time_series_exponential_weighted_moving_average(x_list[4]))
#     fitting_features.extend(time_series_double_exponential_weighted_moving_average(x_list[4]))
#     # fitting_features.extend(time_series_periodic_features(x_list[0], x_list[1], x_list[2], x_list[3], x_list[4]))
#     # append yourself fitting features here...
#
#     return fitting_features

def extract_features_with_param(time_series, window):
    ## type: (object, object) -> object
    ## type: (object, object) -> object
    """
    Extracts three types of features from the time series.
    :param time_series: the time series to extract the feature of
    :type time_series: pandas.Series
    :param window: the length of window
    :type window: int
    :return: the value of features
    :return type: list with float
    """
    # if not tsd_common.is_standard_time_series(time_series, window):
    #     # add your report of this error here...
    #
    #     return []

    # spilt time_series
    split_time_series = tsd_common.split_time_series(time_series, window)
    normalized_split_time_series = tsd_common.normalize_time_series(split_time_series)
    max_min_normalized_time_series = tsd_common.normalize_time_series_by_max_min(split_time_series)

    s_features_with_parameter1 = statistical_features.get_parameters_features(max_min_normalized_time_series)
    # s_features_with_parameter2 = statistical_features.get_parameters_features(normalized_split_time_series)
    features = s_features_with_parameter1
    return features
#######后期做笔记
    # s_features_with_parameter=pd.DataFrame(s_features_with_parameter)
    # s_features_with_parameter_format_changed= []
    # k_format_changed = pd.DataFrame(s_features_with_parameter_format_changed)
    # for i in s_features_with_parameter.columns.tolist():
    #     t = s_features_with_parameter.iloc[:,i].tolist()
    #     y = pd.DataFrame(t)
    #     s_features_with_parameter_format_changed = pd.concat([s_features_with_parameter_format_changed,y],axis=1)






# s_features_with_parameter2 = feature_service.calculate_param_features(total_dataset.value, extract_settings)
    # tmp = _do_extraction_on_chunk(total_dataset.value, extract_settings)


# features = s_features
#     return s_features_with_parameter
#     features = s_features_with_parameter
#     return features

    # return features


def extract_features_without_param(time_series, window):
    ## type: (object, object) -> object
    ## type: (object, object) -> object
    """
    Extracts three types of features from the time series.
    :param time_series: the time series to extract the feature of
    :type time_series: pandas.Series
    :param window: the length of window
    :type window: int
    :return: the value of features
    :return type: list with float
    """
    # if not tsd_common.is_standard_time_series(time_series, window):
    #     # add your report of this error here...
    #
    #     return []

    # spilt time_series
    split_time_series = tsd_common.split_time_series(time_series, window)
    split_time_series2 = tsd_common.split_time_series2(time_series, window)

    # nomalize time_series
    normalized_split_time_series = tsd_common.normalize_time_series(split_time_series)
    max_min_normalized_time_series = tsd_common.normalize_time_series_by_max_min(split_time_series)
    s_features = statistical_features.get_statistical_features(normalized_split_time_series[4])
    f_features = fitting_features.get_fitting_features(normalized_split_time_series)
    c_features = classification_features.get_classification_features(max_min_normalized_time_series)
    # combine features with types
    # s_features_without_parameter = statistical_features.calculate_nonparameters_features(normalized_split_time_series[4])
    # s_features_with_parameter = statistical_features.get_parameters_features(normalized_split_time_series[4])
    # s_features_with_parameter = statistical_features.get_parameters_features(time_series)

    # features = c_features
    #     return s_features_with_parameter
    features = s_features + c_features + f_features
    # features = c_features
    return features

def calculate_all_features(time_series, window):
    ## type: (object, object) -> object
    ## type: (object, object) -> object
    """
    Extracts three types of features from the time series.
    :param time_series: the time series to extract the feature of
    :type time_series: pandas.Series
    :param window: the length of window
    :type window: int
    :return: the value of features
    :return type: list with float
    """
    # if not tsd_common.is_standard_time_series(time_series, window):
    #     # add your report of this error here...
    #
    #     return []

    # spilt time_series
    split_time_series = tsd_common.split_time_series(time_series, window)
    # split_time_series2 = tsd_common.split_time_series2(time_series, window)

    # nomalize time_series
    normalized_split_time_series = tsd_common.normalize_time_series(split_time_series)
    max_min_normalized_time_series = tsd_common.normalize_time_series_by_max_min(split_time_series)
    # s_features = statistical_features.get_statistical_features(normalized_split_time_series[4])
    # c_features = classification_features.get_classification_features(max_min_normalized_time_series)
    # f_features = fitting_features.get_fitting_features(normalized_split_time_series)
    s_features_with_parameter1 = statistical_features.get_parameters_features(max_min_normalized_time_series)

    # features = s_features + c_features + f_features + s_features_with_parameter1
    features = s_features_with_parameter1
    # features = c_features
    return features