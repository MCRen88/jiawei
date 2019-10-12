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
from time_series_detector.common import tsd_common



def extract_features_with_param(time_series, window):
    # type: (object, object) -> object
    # type: (object, object) -> object
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
    # type: (object, object) -> object
    # type: (object, object) -> object
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
    # type: (object, object) -> object
    # type: (object, object) -> object
    """
    Extracts three types of features from the time series.
    :param time_series: the time series to extract the feature of
    :type time_series: pandas.Series
    :param window: the length of window
    :type window: int
    :return: the value of features
    :return type: list with float
    """

    split_time_series = tsd_common.split_time_series(time_series, window)
    # nomalize time_series
    normalized_split_time_series = tsd_common.normalize_time_series(split_time_series)
    max_min_normalized_time_series = tsd_common.normalize_time_series_by_max_min(split_time_series)
    # s_features = statistical_features.get_statistical_features(normalized_split_time_series[4])
    # c_features = classification_features.get_classification_features(max_min_normalized_time_series)
    # f_features = fitting_features.get_fitting_features(normalized_split_time_series)
    s_features_with_parameter1 = statistical_features.get_parameters_features(max_min_normalized_time_series)

    # features = s_features + c_features + f_features + s_features_with_parameter1
    features = s_features_with_parameter1
    return features