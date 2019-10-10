#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import csv
import re
# import matplotlib.pylab as plt

# from inspect import signature
import datetime
import warnings
import time
import json, ast
from datetime import timedelta
import json, ast

# import pywt #导入PyWavelets

# import feature
# import feature.extraction
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import classification_report,confusion_matrix,f1_score,average_precision_score
from sklearn import datasets, svm
from sklearn.feature_selection import chi2,mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold,SelectFromModel
from sklearn.metrics import precision_recall_curve
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif

from statsmodels.stats.diagnostic import unitroot_adf
# import tushare as ts
from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
# from imblearn.over_sampling import SMOTE #数据不均衡
# from scikitplot.classifiers import plot_precision_recall_curve

# from time_series_detector.feature.extraction import *
from time_series_detector.algorithm.gbdt import *
import time_series_detector.algorithm.gbdt
# from sklearn.cross_validation import train_test_split



def sliding_window(value, window_len):
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

def calculate_features_with_param(data, window):

    """
    Caculate time features.

    :param data: the time series to detect of
    :param window: the length of window
    """
    features = []
    label = []
    sliding_arrays = sliding_window(data.value, window_len=window)

    for ith, arr in enumerate(sliding_arrays):
        tmp = feature_service.extract_features_with_param(arr, window)
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

    return [k_format_changed, label]


def calculate_features_without_param(data, window):

    """
    Caculate time features.

    :param data: the time series to detect of
    :param window: the length of window
    """

    features = []
    label = []
    DAY_PNT = len(total_dataset.loc[total_dataset['Date'] == total_dataset['Date'].ix[len(total_dataset)/2]])
    sliding_arrays = sliding_window(data.value, window_len=window)

    for ith, arr in enumerate(sliding_arrays):
        tmp = feature_service.extract_features_without_param(arr, window) ##by shihuan
        features.append(tmp)

    k = pd.DataFrame(features)
    k_format_changed= []
    k_format_changed = pd.DataFrame(k_format_changed)
    for i in k.columns.tolist():
        t = k.iloc[:, i].tolist()
        y = pd.DataFrame(t)
        k_format_changed = pd.concat([k_format_changed, y], axis=1)
    # # x_train format
    k_format_changed = k_format_changed.fillna(0)
    k_format_changed = k_format_changed.ix[:, ~((k_format_changed==0).all())] ##delete the columns that is all 0 values
    k_format_changed = k_format_changed.ix[:, ~((k_format_changed==1).all())] ##delete the columns that is all 1 values

    # if with_label:
    # label = data.anomaly.values[window + 7 * DAY_PNT -1 : len(data.value) + 1]
    label = data.anomaly[window + 7 * DAY_PNT-1:]
    label = pd.DataFrame(label)

    return [k_format_changed, label]


def combine_features_calculate (data, window):#####汇合了metis和tsfresh
    features = []
    label = []
    sliding_arrays = sliding_window(data.value, window_len=window)

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
    # label = pd.DataFrame(label)
    # values = data.values[window + 7 * DAY_PNT-1:]
    # values = pd.DataFrame(values)
    # timestamps = data.timestamps[window + 7 * DAY_PNT-1:]
    # timestamps = pd.DataFrame(timestamps)
    return [k_format_changed, label]





def Precision_Recall_Curve(training_data_anomaly, anomaly_score_train,test_data_anomaly, anomaly_score_test):
    """
    Obtain values of Precision and Recall of the prediction;
    Draw Precision_Recall_Curve

    :param y_true: True binary labels.
                If labels are not either {-1, 1} or {0, 1}, then pos_label should be explicitly given.
    :param y_pred_proba: Estimated probabilities or decision function.
    :param pos_label: The label of the positive class.
    """
    # def Precision_Recall_assessed_value(precision, recall,draw_type)
    # P-R图
    precision_train, recall_train, threshold_train = precision_recall_curve(training_data_anomaly, anomaly_score_train)
    precision_test, recall_test, threshold_test = precision_recall_curve(test_data_anomaly, anomaly_score_test)

    print(precision_train)
    print(recall_train)
    print(threshold_train)
    # plt.plot(recall,precision)
    plt.plot(recall_train,precision_train,label = 'train_precision_recall')
    # plt.plot(recall_test,precision_test,label = 'test_precision_recall')

    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc = 'best')
    plt.show()

#'Precision and Recall of Training Data
    plt.plot(precision_train,label = 'precision_train')
    plt.plot(recall_train,label = 'recall_train')
    plt.title('Precision and Recall of Training Data')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc = 'best')
    plt.show()

#Precision and Recall of Test Data
    plt.plot(precision_test,label = 'precision_test')
    plt.plot(recall_test,label = 'recall_test')
    plt.title('Precision and Recall of Test Data')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc = 'best')
    plt.show()

    return precision_train, recall_train, threshold_train ,precision_test, recall_test, threshold_test

def true_predict_curve(y,x):
    y_pred = clf.predict(x)
    plt.plot(y, color='blue', label='true data')
    plt.plot(y_pred, color='red',label='predict data')
    plt.legend(loc='best')
    plt.title('Comparison of True and Predict Anomaly Curve')
    plt.show(block=False)


#绘制R/P曲线
#原文链接：https://blog.csdn.net/lsldd/article/details/41551797

def plot_pr(auc_score, precision, recall, label=None):
    plot_pr.figure(num=None, figsize=(6, 5))
    plot_pr.xlim([0.0, 1.0])
    plot_pr.ylim([0.0, 1.0])
    # pylplot_prob.xlabel('Recall')
    plot_pr.ylabel('Precision')
    plot_pr.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
    plot_pr.fill_between(recall, precision, alpha=0.5)
    plot_pr.grid(True, linestyle='-', color='0.75')
    plot_pr.plot(recall, precision, lw=1)
    plot_pr.show()

# def comparison_curve():
#先画出原图
#找出与预测不同的值，变成另一个新的dataframe；
#画出这个dataframe--用红色
#plt.show（）

# def Dataset_Plot(dataset_name, color , marker ,label):
#     plt.plot(pd.to_datetime(np.array(dataset_name.timestamps)), dataset_name.value, marker, color, label)
# def Dataset_Scatter(dataset_name, color , marker ,label):
#     plt.plot(pd.to_datetime(np.array(dataset_name.timestamps)), dataset_name.value, marker, color, label)

# def Daytime_Extraction(dataset):
#     dataset_daytime_extraction = time.strftime("%H:%M:%S", time.gmtime(dataset.timestamps))
#     return Dataset_Daytime_Extraction


# def moving_features(df_with_columns_name):
#     ##feature_df is a dataframe including the timeseries,time, add the calculate features into this dataframe.
#     ##the value should be the insert data
#     global feature_df
#     feature_df = total_dataset.value ##the total dataset refer to the data
#     feature_df = pd.DataFrame(feature_df,columns={'value'})
#     for i in range(0,len(df_with_columns_name)):
#         statistic_name = '{}'.format(df_with_columns_name[i])
#         feature_for_add = []
#         for w in range(0,len(total_dataset)):
#             temp = df_with_columns_name[i](feature_df.value[0:w+1])
#             feature_for_add.append(temp)
#         feature_for_add= pd.DataFrame(feature_for_add,columns={'{}'.format(df_with_columns_name[i])})
#         feature_df = pd.concat([feature_df, feature_for_add],axis=1)
#     return feature_df
#


def millisec_to_str(sec):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(sec)))

def day_based_changed(target_date):
    total_dataset_Date_index = total_dataset
    total_dataset_Date_index = total_dataset_Date_index.set_index('Date')
    total_dataset_Date_index.index = pd.DatetimeIndex(total_dataset_Date_index.index)

    line_name = 'Line_{}'.format(target_date)
    line = total_dataset_Date_index.loc[target_date] ##12。24为基准
    plt.plot(line.Hour_Minute, line.value,  label=line_name,alpha = 0.4)
    if len(line.loc[line['anomaly']== 1])>0:
        line_anomaly = line.loc[line['anomaly']== 1]
        plt.scatter(np.array(line_anomaly.Hour_Minute),line_anomaly.value, marker='+',color = tatget_color, label='Line_{}'.format(target_date))
    delta=datetime.timedelta(days=1)
    # yesterday = pd.datetime(target_date) - delta
    line_name2 = 'Line_{}'.format(yesterday)
    line = total_dataset_Date_index.loc[yesterday] ##12。24为基准
    plt.plot(line.Hour_Minute, line.value,label=yesterday,alpha = 0.4)
    if len(line.loc[line['anomaly']== 1])>0:
        line_anomaly = line.loc[line['anomaly']== 1]
        plt.scatter(np.array(line_anomaly.Hour_Minute),line_anomaly.value, marker='+',color = 'blue', label='Line_{}'.format(yesterday))

    plt.title('{} Day-based Comparison (with yesterday and last week)'.format(target_date))
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.xticks(rotation=30)
    plt.show()


##应该修改成自动加减时间
def day_based_changed_proba_predict(target_date1,target_date2, target_date3):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    total_dataset_Date_index = total_dataset
    total_dataset_Date_index = total_dataset_Date_index.set_index('Date')
    total_dataset_Date_index.index = pd.DatetimeIndex(total_dataset_Date_index.index)


    line_name1 = 'Line_{}'.format(target_date1)
    line_name2 = 'Line_{}'.format(target_date2)
    line_name3 = 'Line_{}'.format(target_date3)
    # yesterday = target_date - datetime.timedelta(days = 1).strftime('%Y-%m-%d')
    # lastweek = target_date - datetime.timedelta(days = 7).strftime('%Y-%m-%d')
    line1 = total_dataset_Date_index.loc[target_date1] ##12。24为基准
    line2 = total_dataset_Date_index.loc[target_date2] ##12。24为基准
    line3 = total_dataset_Date_index.loc[target_date3] ##12。24为基准
    ax.plot(line1.Hour_Minute, line1.value, color = 'red',  label=line_name1,alpha = 0.4)
    ax.plot(line2.Hour_Minute, line2.value, color = 'green',  label=line_name2,alpha = 0.4)
    ax.plot(line3.Hour_Minute, line3.value, color = 'blue',  label=line_name3,alpha = 0.4)
    if len(line1.loc[line1['anomaly']== 1])>0:
        line_anomaly = line1.loc[line1['anomaly']== 1]
        ax.scatter(np.array(line_anomaly.Hour_Minute),line_anomaly.value, marker='+',color = 'red', label='{}'.format(line_name1))
    if len(line2.loc[line2['anomaly']== 1])>0:
        line_anomaly = line2.loc[line2['anomaly']== 1]
        ax.scatter(np.array(line_anomaly.Hour_Minute),line_anomaly.value, marker='+',color = 'red', label='{}'.format(line_name2))
    if len(line3.loc[line3['anomaly']== 1])>0:
        line_anomaly = line3.loc[line3['anomaly']== 1]
        ax.scatter(np.array(line_anomaly.Hour_Minute),line_anomaly.value, marker='+',color = 'red', label='{}'.format(line_name3))


    ax2 = ax.twinx()
    # lns3 = ax2.plot(line1.Hour_Minute, line1.y_proba_pred_total, label = 'Anomaly Score')
    # lns = lns1+lns2+lns3
    # labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # ax.legend(bbox_to_anchor=(1.0, 0.05), loc=3, borderaxespad=0)
    ax.legend(loc = 'best')
    ax2.legend(loc = 'best')
    plt.xticks(rotation=30)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Value changed")
    # ax2.set_ylabel("Probability Anomaly Score")
    # ax2.set_ylim(0, 1)
    plt.xticks(rotation = 30)
    plt.title('{} Day-based Comparison (with yesterday and last week)'.format(target_date1))

    plt.show()




def anomaly_score_view_date(dataset,target_date):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    total_dataset_Date_index = dataset
    total_dataset_Date_index = total_dataset_Date_index.set_index('Date')
    total_dataset_Date_index.index = pd.DatetimeIndex(total_dataset_Date_index.index)


    line_name1 = 'Line_{}'.format(target_date)
    line1 = total_dataset_Date_index.loc[target_date] ##12。24为基准
    ax.plot(line1.Hour_Minute, line1.value, color = 'red',  label=line_name1,alpha = 0.4)
    if len(line1.loc[line1['anomaly']== 1])>0:
        line_anomaly = line1.loc[line1['anomaly']== 1]
        ax.scatter(np.array(line_anomaly.Hour_Minute),line_anomaly.value, marker='+',color = 'red', label='{}'.format(line_name1))

    ax2 = ax.twinx()
    ax2.plot(line1.Hour_Minute, line1.anomaly_pred_score, label = 'Anomaly Score')
    ax.legend(loc = 'best')
    ax2.legend(loc = 'best')
    plt.xticks(rotation=30)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Value changed")
    ax2.set_ylabel("Probability Anomaly Score")
    ax2.set_ylim(0, 1)
    plt.xticks(rotation = 30)
    plt.title('Anomaly Score Date View')
    plt.show()

def anomaly_score_view_predict(dataset):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    anomaly_flag = dataset.set_index("anomaly")
    anomaly_flag = anomaly_flag.loc[1]

    plt.plot(pd.to_datetime(dataset.timestamps),dataset.value, color = 'black',alpha = 0.3)
    plt.scatter(pd.to_datetime(np.array(anomaly_flag.timestamps)),anomaly_flag.value, color='black',marker='+', label='Anomaly_Flag',alpha = 0.8)
    plt.xticks(rotation=30)

    ax2 = ax.twinx()
    ax2.plot(pd.to_datetime(dataset.timestamps), dataset.anomaly_pred_score, label = 'Anomaly Score')
    ax.legend(loc = 'best')
    ax2.legend(loc = 'best')

    ax.set_xlabel("Date (h)")
    ax.set_ylabel("Value changed")
    ax2.set_ylabel("Probability Anomaly Score")
    ax2.set_ylim(0, 1)
    plt.title('Anomaly Score Predict View')

    plt.show()


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

def data_modeling_gbdt(x_selected,y):
    '''
    This func utilizing gbdt to model the x_selected and Label data, and predict the percentage of anomaly and also,
    the results of anomaly. It is worth to know that there has the threshould to distinct to what percentage the results should
    be regarded as anomaly.

    :param x_selected: selected features dataset
    :param y: Label data (or Flag)
    :return: y_pred(not DF), represents whether the predicte value would be 0 or 1(anomaly)
     anomaly_score(not DF), represents the percentage to be anomaly.
    '''

    clf = GradientBoostingClassifier()
    clf.fit(x_selected, y)
    # for cnt, tree in enumerate(clf.estimators_):
    #     plot_tree(clf=tree[0], title="example_tree_%d" % cnt)

    y_pred = clf.predict(x_selected)
    anomaly_score = clf.predict_proba(x_selected)[:, 1]
    # y_pred = pd.DataFrame(y_pred)
    # anomaly_score = pd.DataFrame(anomaly_score)
    return y_pred, anomaly_score

def anomaly_view(dataset):
    y_train_anomaly_dataset_selcted = dataset.set_index("anomaly")
    y_train_anomaly_dataset_selcted = y_train_anomaly_dataset_selcted.loc[1]

    plt.plot(pd.to_datetime(total_dataset.timestamps),total_dataset.value, color = 'black',alpha = 0.3)
    plt.scatter(pd.to_datetime(np.array(y_train_anomaly_dataset_selcted.timestamps)),y_train_anomaly_dataset_selcted.value, color='red',marker='+', label='Maxmin_value_anomaly')
    plt.legend(loc= 'best',fontsize= 5)
    plt.xticks(rotation=30)
    plt.title('Value and Anomaly view')# give plot a title
    plt.show()

def anomaly_predict_view(dataset):
    anomaly_flag = dataset.set_index("anomaly")
    anomaly_flag = anomaly_flag.loc[1]
    anomaly_pridict = dataset.set_index("anomaly_pred")
    anomaly_pridict = anomaly_pridict.loc[1]

    plt.plot(pd.to_datetime(dataset.timestamps),dataset.value, color = 'black',alpha = 0.3)
    plt.scatter(pd.to_datetime(np.array(anomaly_flag.timestamps)),anomaly_flag.value, color='black',marker='+', label='Anomaly_Flag',alpha = 0.8)
    plt.scatter(pd.to_datetime(np.array(anomaly_pridict.timestamps)),anomaly_pridict.value, color='orange',marker='^', label='Anomaly_Predict',alpha = 0.8)

    plt.legend(loc= 'best',fontsize= 5)
    plt.xticks(rotation=30)
    plt.title('Anomaly Predict View')

    plt.show()

def circulation_file_predict_origin_features_select_methods(total_dataset):
    #思路
    ##先进行绘图：1、曲线和异常点的图；2、观察某个随机日期的数据（自行输入日期），得到前一天；上一周和本周的异常情况
    ##先进行特征计算
    #再进行特征选择
    #在进行数据集 split；从win+7day开始--从win+7day开始的数据集
    #在进行预测算法；
    #对异常的预测情况的precision_recall的画图；
    #单独的precision和recall图
    #结果根据时间窗添加到原始total——dataset中,对比flag为异常和预测为异常的情况


    ##先进行绘图：1、曲线和异常点的图；2、观察某个随机日期的数据（自行输入日期），得到前一天；上一周和本周的异常情况
    #1、曲线和异常点的图

    anomaly_view(total_dataset)#观察异常点和整体时序走向
    day_based_changed_proba_predict('2015-01-07','2015-01-06','2014-12-31')


##先进行特征计算
    x_features_calculate,y_calculate = combine_features_calculate (total_dataset, window)

    #再进行特征选择
    x_features_selected = features_selected_ExtraTreesClassifier(x_features_calculate, y_calculate)

    selected_features_name, x_features_selected = selected_columns_names(x_features_calculate, x_features_selected)
    # print x_features_selected.columns.tolist()

    #在进行数据集 split；从win+7day开始--从win+7day开始的数据集
    new_dataset = total_dataset.iloc[win_sli-1:,:]
    new_dataset = new_dataset.reset_index(drop= True)
    training_data, test_data = train_test_split(new_dataset, test_size = 0.3, shuffle=False)
    x_train_selected, x_test_selected = train_test_split(x_features_selected, test_size = 0.3, shuffle=False)
    y_train_selected,y_test_selected = train_test_split(y_calculate, test_size = 0.3, shuffle=False)




    # #在进行预测算法；
    y_pred_train, anomaly_score_train = data_modeling_gbdt(x_train_selected,training_data.anomaly)
    y_pred_test, anomaly_score_test = data_modeling_gbdt(x_test_selected,test_data.anomaly)

    predict_report_train = classification_report (training_data.anomaly,y_pred_train, labels=[1, 2, 3])
    predict_report_test = classification_report (test_data.anomaly,y_pred_test, labels=[1, 2, 3])

    # print '\n\npredict_report_train\n\n',predict_report_train,'\n\npredict_report_test\n\n\n',predict_report_test

    # #结果根据时间窗添加到原始total——dataset中
    print "\n\ntraining_data\n",training_data,"\n\ntest_data\n",test_data,"\n\n\nnew_dataset\n",new_dataset


    anomaly_pred = []
    anomaly_pred.extend(y_pred_train)
    anomaly_pred.extend(y_pred_test)
    anomaly_pred = pd.DataFrame(anomaly_pred,columns={'anomaly_pred'})


    anomaly_pred_score = []
    anomaly_pred_score.extend(anomaly_score_train)
    anomaly_pred_score.extend(anomaly_score_test)
    anomaly_pred_score = pd.DataFrame(anomaly_pred_score,columns={'anomaly_pred_score'})
    #
    new_dataset = pd.concat([new_dataset,anomaly_pred,anomaly_pred_score],axis=1)  #new dataset 包含了的数据包括原始数据集的属性+预测值和预测分数；并ignore window+7*day_pnt的值


    anomaly_predict_view(new_dataset) ##compare the difference between anomaly_flag and anomaly_predict
    anomaly_score_view_predict(new_dataset) ##观察整条曲线的anomaly_score和anomaly——flag的情况

    anomaly_score_view_date(new_dataset,'2015-01-07') ##观察某一天anomaly_score和anomaly——flag的情况

# #对异常的预测情况的precision_recall的画图；
    Precision_Recall_Curve(training_data.anomaly, anomaly_score_train,test_data.anomaly, anomaly_score_test)

    # #单独的precision和recall图


##平稳性测试
def adf_(timeseries): # adf_ 检验平稳性
    adf_test = unitroot_adf(timeseries)
    adf_test_value = adf_test[0]
    adfuller_value = pd.DataFrame({key:value for key,value in adf_test[4].items()},index = [0])
    adfuller_value = pd.DataFrame(adfuller_value)
    adfuller_critical_value = adfuller_value['10%'][0]
    return adf_test_value, adfuller_critical_value


def kpss_(timeseries): #kpss检验平稳性
    kpss_test = kpss(timeseries)
    kpss_test_value = kpss_test[0]
    kpss_value = pd.DataFrame({key:value for key,value in kpss_test[3].items()},index = [0])
    kpss_value = pd.DataFrame(kpss_value)
    kpss_critical_value = kpss_value['10%'][0]
    return kpss_test_value, kpss_critical_value

def acorr_ljungbox_(timeseries):
    a = acorr_ljungbox(timeseries, lags=1)
    return a[1][0] ### return 检验结果的 p_value值


# def smooth_and_nonrandom_test(dataset):
#
#     adf_test_value, adf_critical_value = adf_(dataset.value)
#     kpss_test_value, kpss_critical_value = kpss_(dataset.value)
#     #
#     if adf_test_value > adf_critical_value and kpss_test_value > kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
#         print '原始序列绝对不平稳'
#         dataset["value_diff"] = dataset["value"] - dataset["value"].shift(1)
#         ######
#         dataset = dataset.ix[1:].reset_index()
#
#         adf_test_value2, adf_critical_value2 = adf_(dataset.value_diff)
#         kpss_test_value2, kpss_critical_value2 = kpss_(dataset.value_diff)
#         if adf_test_value2 < adf_critical_value2 and kpss_test_value2 < kpss_critical_value2:
#             print '一阶差分后平稳'
#             a = acorr_ljungbox_(dataset.value_diff)
#             if a < 0.05:
#                 dataset.value = dataset.value_diff ###由于差分之后时间序列平稳，用差分后的值代替value进行分析；
#                 list_r = circulation_file_predict_origin_features_select_methods(dataset)
#
#     ######需要一个能够自动判断不平稳数据后经过在多次处理后 再循环判断，一直到数据平稳的过程
#
#
#
#
#
#     # if adf_test_value > adf_critical_value and kpss_test_value < kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
#     #     print '趋势平稳--去除趋势后序列严格平稳'
#     #
#     # if adf_test_value < adf_critical_value and kpss_test_value > kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
#     #     print '差分平稳，利用差分可使序列平稳'
#     #     dataset["value_diff"] = dataset["value"] - dataset["value"].shift(1)
#     #     adf_test2 =  unitroot_adf(dataset.value_diff)
#     #     # adfuller_value2= pd.DataFrame({key:value for key,value in adf_test[4].items()},index = [0])
#     #     # adfuller_value = pd.DataFrame(adfuller_value)
#     #     print adf_test2
#     #
#     # elif adf_test_value < adf_critical_value and kpss_test_value < kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
#     #     print '绝对平稳'
#     #     list_r = circulation_file_predict_origin_features_select_methods(dataset)
#     #     list_to_print.append(list_r)
#     # print kpss(dataset.value)
#
#
#



##平稳性测试
def adf_(timeseries): # adf_ 检验平稳性
    adf_test = unitroot_adf(timeseries)
    adf_test_value = adf_test[0]
    adfuller_value = pd.DataFrame({key:value for key,value in adf_test[4].items()},index = [0])
    adfuller_value = pd.DataFrame(adfuller_value)
    adfuller_critical_value = adfuller_value['10%'][0]
    return adf_test_value, adfuller_critical_value


def kpss_(timeseries): #kpss检验平稳性
    kpss_test = kpss(timeseries)
    kpss_test_value = kpss_test[0]
    kpss_value = pd.DataFrame({key:value for key,value in kpss_test[3].items()},index = [0])
    kpss_value = pd.DataFrame(kpss_value)
    kpss_critical_value = kpss_value['10%'][0]
    return kpss_test_value, kpss_critical_value

def acorr_ljungbox_(timeseries):
    a = acorr_ljungbox(timeseries, lags=1)
    return a[1][0] ### return 检验结果的 p_value值


# def smooth_and_nonrandom_test(dataset):
#
#     adf_test_value, adf_critical_value = adf_(dataset.value)
#     kpss_test_value, kpss_critical_value = kpss_(dataset.value)
#     #
#     if adf_test_value > adf_critical_value and kpss_test_value > kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
#         print '原始序列绝对不平稳'
#         dataset["value_diff"] = dataset["value"] - dataset["value"].shift(1)
#         ######
#         dataset = dataset.ix[1:].reset_index()
#
#         adf_test_value2, adf_critical_value2 = adf_(dataset.value_diff)
#         kpss_test_value2, kpss_critical_value2 = kpss_(dataset.value_diff)
#         if adf_test_value2 < adf_critical_value2 and kpss_test_value2 < kpss_critical_value2:
#             print '一阶差分后平稳'
#             a = acorr_ljungbox_(dataset.value_diff)
#             if a < 0.05:
#                 dataset.value = dataset.value_diff ###由于差分之后时间序列平稳，用差分后的值代替value进行分析；
#                 list_r = circulation_file_predict_origin_features_select_methods(dataset)
#
#     ######需要一个能够自动判断不平稳数据后经过在多次处理后 再循环判断，一直到数据平稳的过程
#
#
#
#
#
#     # if adf_test_value > adf_critical_value and kpss_test_value < kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
#     #     print '趋势平稳--去除趋势后序列严格平稳'
#     #
#     # if adf_test_value < adf_critical_value and kpss_test_value > kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
#     #     print '差分平稳，利用差分可使序列平稳'
#     #     dataset["value_diff"] = dataset["value"] - dataset["value"].shift(1)
#     #     adf_test2 =  unitroot_adf(dataset.value_diff)
#     #     # adfuller_value2= pd.DataFrame({key:value for key,value in adf_test[4].items()},index = [0])
#     #     # adfuller_value = pd.DataFrame(adfuller_value)
#     #     print adf_test2
#     #
#     # elif adf_test_value < adf_critical_value and kpss_test_value < kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
#     #     print '绝对平稳'
#     #     list_r = circulation_file_predict_origin_features_select_methods(dataset)
#     #     list_to_print.append(list_r)
#     # print kpss(dataset.value)
#
#
#


from sklearn.tree import export_graphviz
import graphviz


def plot_tree(clf, title="example"):
    from sklearn.tree import export_graphviz
    import graphviz
    dot_data = export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(title)
    pass







##########################################--- main ----#####################################################################
##########################################--- main ----#####################################################################
if __name__ == "__main__":


    global x_features_selected

    warnings.filterwarnings("ignore")
    window = 14
    DEFAULT_WINDOW = 14
    k = pd.read_csv('data/ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/totoal_file_and_name.csv')
    k = pd.DataFrame(k)
    list_to_print = []
    for i in range(0,len(k)):
        location = k["location"].ix[i]
        filename = k["filename"].ix[i]
        total_dataset = pd.read_csv('{}'.format(location))
        if i >= 124:
            total_dataset.rename(columns={"timestamp":"timestamps", "is_anomaly":"anomaly"}, inplace=True)
        ##--1。时间处理格式，顺序--##
        total_dataset['timestamps'] = total_dataset['timestamps'].map(millisec_to_str)
        total_dataset['timestamps'] = total_dataset['timestamps'].map(pd.to_datetime)
        total_dataset = total_dataset.sort_values(by=['timestamps'], ascending=True)

        ##--2。判断确实值并对缺失数据进行填充，目标只针对timestamps,value,anomaly--##--
        if total_dataset.timestamps.isnull().any():
            total_dataset.dropna(subset=['timestamps'])
        if total_dataset.anomaly.isnull().any():
            total_dataset.dropna(subset=['anomaly'])
        if total_dataset.value.isnull().any():
            total_dataset.fillna(total_dataset.value.mean())

        # #数据拆分和合成新列
        #   #时间数据拆分--根据timestamps添加2个新列--Date和Hour_Minute
        total_dataset['Hour_Minute'] = total_dataset['timestamps'].dt.time
        total_dataset['Date'] = total_dataset['timestamps'].dt.date

        #     #########total_dataset中含有两新列----Date和Hour_Minute
        DAY_PNT = len(total_dataset.loc[total_dataset['Date'] == total_dataset['Date'].ix[len(total_dataset)/2]])

        lenth_total_dataset = len(total_dataset)
        win_sli = window + 7 * DAY_PNT
        lenth_new_dataset = len(total_dataset.ix[win_sli-1:]) #真正有特征值部分的数据集

        training_data, test_data = train_test_split(total_dataset.ix[win_sli-1:], test_size = 0.3, shuffle=False)
        train_ = total_dataset.ix[win_sli-1:int(lenth_new_dataset*0.7)+win_sli-1]
        test_ = total_dataset.ix[int(lenth_new_dataset*0.7)+win_sli-1:]

        # plt.plot(total_dataset.value)
        # plt.legend(loc= 'best',fontsize= 5)
        # plt.xticks(rotation=30)
        # plt.title('Value view')# give plot a title
        # plt.show()
        #break
        #
        # #
        if win_sli < int(len(total_dataset)*0.3) and len(test_.loc[test_['anomaly'] == 1]) > 0 and len(train_.loc[train_['anomaly'] == 1]) > 0: ####（要加入判断时间序列的分析有没有价值的判断方法）
            print total_dataset.head(3)
            total_dataset.to_csv("for_look.csv")
            anomaly_view(total_dataset)#观察异常点和整体时序走向 （未进行数据平稳处理前）
            # #判断序列的平稳性
            # plt.stem(stattools.acf(total_dataset.value))
            # plt.stem(stattools.pacf(total_dataset.value))
            # plt.show()


            # # by values
            adf_test_value, adf_critical_value = adf_(total_dataset.value)
            kpss_test_value, kpss_critical_value = kpss_(total_dataset.value)
            # a = acorr_ljungbox(total_dataset.value, lags=1)
            adf_test = unitroot_adf(total_dataset.value)
            kpss_test = kpss(total_dataset.value)

            # print '\nacorr_ljungbox\n',a
            print '\nunitroot_adf',adf_test
            print '\nkpss\n',kpss_test

            #
            if adf_test_value > adf_critical_value and kpss_test_value > kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
                print '原始序列绝对不平稳'
                total_dataset["value_diff"] = total_dataset["value"] - total_dataset["value"].shift(1)
                ######


                total_dataset = total_dataset.ix[1:].reset_index()
                adf_test_value2, adf_critical_value2 = adf_(total_dataset.value_diff)
                kpss_test_value2, kpss_critical_value2 = kpss_(total_dataset.value_diff)


                a2 = acorr_ljungbox(total_dataset.value, lags=1)
                adf_test2 = unitroot_adf(total_dataset.value)
                kpss_test2 = kpss(total_dataset.value)
            #
                print '\nacorr_ljungbox\n',a2
                print '\nunitroot_adf',adf_test2
                print '\nkpss\n',kpss_test2
            #
            #
            #
            #
                if adf_test_value2 < adf_critical_value2 and kpss_test_value2 < kpss_critical_value2:
                    print '一阶差分后平稳'
                    a = acorr_ljungbox_(total_dataset.value_diff)
                    if a < 0.05:
                        total_dataset.value = total_dataset.value_diff ###由于差分之后时间序列平稳，用差分后的值代替value进行分析；
                        list_r = circulation_file_predict_origin_features_select_methods(total_dataset)
                    else: break
            #
            # ######补充一个能够自动判断不平稳数据后经过在多次处理后 再循环判断，一直到数据平稳的过程
            #
            #
            #
            # # if adf_test_value > adf_critical_value and kpss_test_value < kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
            # #     print '趋势平稳--去除趋势后序列严格平稳'
            # #
            # # if adf_test_value < adf_critical_value and kpss_test_value > kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
            # #     print '差分平稳，利用差分可使序列平稳'
            # #     total_dataset["value_diff"] = total_dataset["value"] - total_dataset["value"].shift(1)
            # #     adf_test2 =  unitroot_adf(total_dataset.value_diff)
            # #     # adfuller_value2= pd.DataFrame({key:value for key,value in adf_test[4].items()},index = [0])
            # #     # adfuller_value = pd.DataFrame(adfuller_value)
            # #     print adf_test2
            # #
            # elif adf_test_value < adf_critical_value and kpss_test_value < kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
            #     print '绝对平稳'
            #     a = acorr_ljungbox_(total_dataset.value_diff)
            #     if a < 0.05:
            #         list_r = circulation_file_predict_origin_features_select_methods(total_dataset)
            #     else: break
            #
            #

            break ##只跑一个数据集
            #
            #
