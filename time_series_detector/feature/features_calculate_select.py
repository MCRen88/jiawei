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
import feature_service
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold,SelectFromModel


def sliding_window(value, window_len,DAY_PNT):
    value_window = []
    value = np.array(value)
    # DAY_PNT = floor(len(value)/7)
    # DAY_PNT = 24
    # DAY_PNT = len(total_dataset.loc[total_dataset['Date'] == total_dataset['Date'].ix[len(total_dataset)/2]])
    for i in range(window_len + 7 * DAY_PNT, len(value) + 1):
        xs_c = value[i - window_len - 7 * DAY_PNT: i + window_len - 7 * DAY_PNT]
        xs_b = value[i - window_len - 1 * DAY_PNT: i + window_len - 1 * DAY_PNT]
        xs_a = value[i - window_len:i]
        xs_tmp = list(xs_c) + list(xs_b) + list(xs_a)
        value_window.append(xs_tmp)

    return value_window
    
def combine_features_calculate (data, window):#####汇合了metis和tsfresh
    features = []
    label = []
    DAY_PNT = len(data.loc[data['Date'] == data['Date'].ix[len(data)/2]])
    sliding_arrays = sliding_window(data.value, window_len=window,DAY_PNT = DAY_PNT)

    for ith, arr in enumerate(sliding_arrays):
        tmp = feature_service.calculate_all_features(arr, window)
        features.append(tmp)

    k = pd.DataFrame(features)
    k_format_changed= []
    k_format_changed = pd.DataFrame(k_format_changed)
    for i in k.columns.tolist():
        t = k.iloc[:, i].tolist()
        y = pd.DataFrame(t)
        k_format_changed = pd.concat([k_format_changed, y], axis=1)
    # # x_train format
    k_format_changed = k_format_changed.astype('float64')
    k_format_changed = k_format_changed.fillna(0)
    k_format_changed = k_format_changed.ix[:, ~((k_format_changed==0).all())] ##delete the columns that is all 0 values
    k_format_changed = k_format_changed.ix[:, ~((k_format_changed==1).all())] ##delete the columns that is all 1 values


    # if with_label:
    label = data.anomaly[window + 7 * DAY_PNT-1:]
    label = pd.DataFrame(label)
    # values = data.values[window + 7 * DAY_PNT-1:]
    # values = pd.DataFrame(values)
    # timestamps = data.timestamps[window + 7 * DAY_PNT-1:]
    # timestamps = pd.DataFrame(timestamps)
    return [k_format_changed, label]

def features_selected_ExtraTreesClassifier(x,y):
    '''
    This func utilizes ExtraTreesClassifier in order to select meaningful features from the calculated features

    :param x: Selected features dataset
    :param y: Label data
    :return: dataset(DF) of selected features
    '''
    clf = ExtraTreesClassifier()
    clf = clf.fit(x, y)
    features_importance = clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    selected_features = model.transform(x)
    selected_features = pd.DataFrame(selected_features)
    return selected_features


def selected_columns_names(origin_features_df, selected_features_df):
    '''
    This func aims to obtain the selected features' names

    :param origin_features_df: the set calculating numbers of values
    :param selected_features_df: the selected features dataset
    :return: name list of selected features
    '''
    k= origin_features_df.columns.tolist()
    k = pd.DataFrame(k)
    len_df = len(selected_features_df) ## The lenth of selected_features_df is the same as that of the origin_features_df
    l1,l2,l3,l4,l5 = int(len_df/2),int(len_df/3),int(len_df/4),int(len_df/5),int(len_df/6)
    l6,l7,l8,l9,l10 = int(len_df/7),int(len_df/8),int(len_df/9),int(len_df/11),len_df-3
    selected_features_name = []

    ##by comparing the random value of two df to find out in which column in origin_features_df is selected and,
    #   therefore, is able to print the sekected features name
    for a in range(0, selected_features_df.shape[1]):
        for b in range(0, origin_features_df.shape[1]):
            if (selected_features_df.iloc[l1, a] == origin_features_df.iloc[l1, b] and selected_features_df.iloc[l2, a] == origin_features_df.iloc[l2, b]
                    and selected_features_df.iloc[l3, a] == origin_features_df.iloc[l3, b] and selected_features_df.iloc[l4, a] == origin_features_df.iloc[l4, b]
                    and selected_features_df.iloc[l5, a] == origin_features_df.iloc[l5, b] and selected_features_df.iloc[l6, a] == origin_features_df.iloc[l6, b]
                    and selected_features_df.iloc[l7, a] == origin_features_df.iloc[l7, b] and selected_features_df.iloc[l8, a] == origin_features_df.iloc[l8, b]
                    and selected_features_df.iloc[l9, a] == origin_features_df.iloc[l9, b] and selected_features_df.iloc[l10, a] == origin_features_df.iloc[l10, b]
            ):
                selected_features_name.append(k.ix[b])
                break

    selected_features_df.columns = np.array(selected_features_name)
    selected_features_name = pd.DataFrame(selected_features_name)
    selected_features_name = selected_features_name.values.tolist() ####！！！！！！！

    return selected_features_name,selected_features_df
    

def feature_extraction(total_dataset,window):
    ##先进行特征计算
    x_features_calculate,y_calculate = combine_features_calculate (total_dataset, window)
    #再进行特征选择
    x_features_selected = features_selected_ExtraTreesClassifier(x_features_calculate, y_calculate)
    selected_features_name, x_features_selected = selected_columns_names(x_features_calculate, x_features_selected)
    # print x_features_selected.columns.tolist()
    return x_features_selected, y_calculate, selected_features_name
