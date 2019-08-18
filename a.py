
# coding: utf-8

import numpy as np
import csv
import pandas as pd
import re
# import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import time
import datetime
from matplotlib.dates import AutoDateLocator, DateFormatter
import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator

import tsfresh.feature_extraction.feature_calculators as ts_feature_calculators
from scipy.cluster.vq import whiten
from sklearn.metrics import f1_score,precision_score
from sklearn.model_selection import train_test_split
from numpy import column_stack
# from imblearn.over_sampling import SMOTE #数据不均衡
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report,confusion_matrix
# from scikitplot.classifiers import plot_precision_recall_curve
from datetime import datetime
import time_series_detector.feature.fitting_features
import tsfresh.feature_extraction.feature_calculators as ts_feature_calculators
from matplotlib.pylab import rcParams

# import feature
from time_series_detector.feature.statistical_features import *
from time_series_detector.feature.classification_features import *
from time_series_detector.feature.feature_service import *
from time_series_detector.feature.fitting_features import *
from time_series_detector.feature.statistical_features import time_series_mean, time_series_variance, time_series_standard_deviation, time_series_median
from time_series_detector.algorithm.gbdt import *
from time_series_detector.common.tsd_common import split_time_series
from time_series_detector.common.tsd_common import normalize_time_series_by_max_min
import operator

def calculate_features(data, window, with_label=True):
    """
    Caculate time features.

    :param data: the time series to detect of
    :param window: the length of window
    """

    features = []
    sliding_arrays = sliding_window(data.value, window_len=window)
    for ith, arr in enumerate(sliding_arrays):
        tmp = feature_service.extract_features(arr, window)
        features.append(tmp)
    if with_label:
        label = data.anomaly.values[window + 7 * 140 - 1:]
    else:
        label = None
    return [features, label]


def Precision_Recall_Curve(y_train, x_train):
    """
    Obtain values of Precision and Recall of the prediction;
    Draw Precision_Recall_Curve

    :param y_true: True binary labels.
                If labels are not either {-1, 1} or {0, 1}, then pos_label should be explicitly given.
    :param y_pred_proba: Estimated probabilities or decision function.
    :param pos_label: The label of the positive class.
    """
    anomaly_score_train = clf.predict_proba(x_train)[:, 1]  # y_train 取值为1（异常）的概率
    precision, recall, thresholds = precision_recall_curve(y_train, anomaly_score_train, pos_label=1)
    # P-R图
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()
    return precision, recall, thresholds

def true_predict_curve(y,x):
    y_pred = clf.predict(x)
    plt.plot(y, color='blue', label='true data')
    plt.plot(y_pred, color='red',label='predict data')
    plt.legend(loc='best')
    plt.title('Comparison of true and predict curve')
    plt.show(block=False)


#绘制R/P曲线
#原文链接：https://blog.csdn.net/lsldd/article/details/41551797

def plot_pr(auc_score, precision, recall, label=None):
    plot_pr.figure(num=None, figsize=(6, 5))
    plot_pr.xlim([0.0, 1.0])
    plot_pr.ylim([0.0, 1.0])
    pylplot_prab.xlabel('Recall')
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

def Dataset_Plot(dataset_name, color , marker ,label):
    plt.plot(pd.to_datetime(np.array(dataset_name.timestamp)), dataset_name.value, marker, color, label)
def Dataset_Scatter(dataset_name, color , marker ,label):
    plt.plot(pd.to_datetime(np.array(dataset_name.timestamp)), dataset_name.value, marker, color, label)

# def Daytime_Extraction(dataset):
#     dataset_daytime_extraction = time.strftime("%H:%M:%S", time.gmtime(dataset.timestamp))
#     return Dataset_Daytime_Extraction




##############--- main ----##################
if __name__ == "__main__":
    window = 60
    total_dataset = pd.read_csv('data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv')

    #数据切分
    training_data, test_data = train_test_split(total_dataset, test_size = 0.3, shuffle=False)
    x_train, y_train = calculate_features(training_data, window)
    x_test, y_test = calculate_features(test_data, window)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)


    #数据模型
    clf = GradientBoostingClassifier(n_estimators=5)
    clf.fit(x_train, y_train)



    y_pred_train= clf.predict(x_train)
    y_pred_test = clf.predict(x_test)



    # print pd.DataFrame(y_train.loc[y_train == 1])




    # y_pred_train = clf.predict()

    # print pd.DataFrame(total_dataset.loc[total_dataset['anomaly']== 0])

    # y_pred_train = clf.predict(np.array(training_data.value))
    # y_pred_test = clf.predict(x_test)
    #!!!!!!!!!!!R curve precision, recall, thresholds = Precision_Recall_Curve(y_train,x_train) # P_R-Curve

    anomaly_dataset = total_dataset.loc[total_dataset['anomaly']== 1]
    # print training_data[-100:]
    # addpred_training_data = training_data
    # addpred_training_data['y_train_pred'] = pd.DataFrame(y_pred_train)
    # addpred_training_data = pd.concat([training_data, pd.DataFrame(y_pred_train)])
    # print pd.DataFrame(x_train,"\n\n\n")
    # print pd.DataFrame(test_data.loc[training_data['anomaly'] == 0])
    # print pd.DataFrame(test_data.loc[training_data['anomaly'] == 1])
    # print pd.DataFrame(y_pred_train)


    #
    #
    # print pd.DataFrame(x_test,"\n\n\n")
    # print pd.DataFrame(test_data.loc[test_data['anomaly']== 1])




    # pred_anomaly_dataset = training_data.loc[training_data['anomaly']== 1]


    #图中需要涉及的内容

    #加入新列，用于放预测是否诗异常的值
    df = pd.DataFrame(columns = ['y_pre_total'])
    add_all_character = total_dataset
    add_all_character = pd.concat([total_dataset,df])
    add_all_character.y_pre_total = add_all_character.y_pre_total.fillna(0)

    #创建预测值的dataframe
    y_pred_train_df = pd.DataFrame(y_pred_train)
    y_pred_train_df.columns = ['y_pred_train']
    y_pred_test_df = pd.DataFrame(y_pred_test)
    y_pred_test_df.columns = ['y_pred_test']

    #窗口长度
    training_data_lenth = int(0.7 * len(total_dataset)) #整个测试集的大小
    window_test_range = window + 7* DAY_PNT #对于每个测试集所抽取 e d

    #加入training
    for i in range(window_test_range,training_data_lenth):
        add_all_character.ix[i,3] = y_pred_train_df.ix[i-window_test_range,0]

    # for i in range((training_data_lenth + window_test_range), len(total_dataset)):
    #     add_all_character.ix[i,3] = y_pred_test_df.ix[i-window_test_range,0]

    #提取出有问异常的数据
    add_all_character_anomaly = add_all_character.loc[add_all_character['y_pre_total'] == 1]
    #!!!!!
    # plt.scatter(pd.to_datetime(np.array(add_all_character_anomaly.timestamp)), add_all_character_anomaly.value, color='black', marker='>',label='Anomaly Value of CPU Utilization - Predicted')
    # plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)), total_dataset.value, color='blue', label='Value of CPU Utilization')
    # plt.scatter(pd.to_datetime(np.array(anomaly_dataset.timestamp)), anomaly_dataset.value, color='red', label='Anomaly Value of CPU Utilization - Flag')

    # plt.legend(loc='best')
    # plt.xticks(rotation=30)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!  不能删 @    plt.show()


    #周期性--一天为周期的图
    time_extraction_df = pd.DataFrame(add_all_character.timestamp)
    time_extraction_df.rename(columns={'timestamp':'Hour_Minute'},inplace=True)
    date_extraction_df = pd.DataFrame(add_all_character.timestamp)
    date_extraction_df.rename(columns={'timestamp':'Date'},inplace=True)
    time_extraction_df['Hour_Minute']= pd.to_datetime(time_extraction_df['Hour_Minute'], format='%Y-%m-%d %H:%M:%S')
    time_extraction_df['Hour_Minute'] = time_extraction_df['Hour_Minute'].dt.time
    date_extraction_df['Date']= pd.to_datetime(date_extraction_df['Date'], format='%Y-%m-%d %H:%M:%S')
    date_extraction_df['Date'] = date_extraction_df['Date'].dt.date
    add_all_character = pd.concat([add_all_character,date_extraction_df,time_extraction_df],axis= 1)
    # print add_all_character.head(3)
    add_all_character['Date']=pd.to_datetime(add_all_character['Date'])

    # print add_all_character.head(100)

    add_all_character = add_all_character.set_index('Date')

    line_2014_02_17 = add_all_character.loc['2014-02-17']
    line_2014_02_18 = add_all_character.loc['2014-02-18']
    line_2014_02_24 = add_all_character.loc['2014-02-24']
    line_2014_02_23 = add_all_character.loc['2014-02-23']

    # print line_2014_02_26


    # !!!!!!!!line_2014_02_15
    plt.plot(line_2014_02_17.Hour_Minute, line_2014_02_17.value, color = 'orange', label='line_2014_02_17',alpha = 0.4)
    if len(line_2014_02_17.loc[line_2014_02_17['anomaly']== 1])>0:
        line_2014_02_17_anomaly = line_2014_02_17.loc[line_2014_02_17['anomaly']== 1]
        lplt.scatter(np.array(line_2014_02_17_anomaly.Hour_Minute),line_2014_02_17_anomaly.value, marker='+',color = 'orange', label='line_2014_02_17_anomaly_flag')


    # if len(line_2014_02_19_anomaly)>0:
    #     plt.scatter(np.array(line_2014_02_19_anomaly.Hour_Minute),line_2014_02_19_anomaly.value, marker='>',color = 'red', label='line_2014_02_15_anomaly')
    # return plt.show()



    #    # !!!!!!!!line_2014_02_19
    plt.plot(line_2014_02_18.Hour_Minute, line_2014_02_18.value, color = 'c', label='line_2014_02_18',alpha = 0.4)
    if len(line_2014_02_18.loc[line_2014_02_18['anomaly']== 1])>0:
        line_2014_02_18_anomaly = line_2014_02_18.loc[line_2014_02_18['anomaly']== 1]
        plt.scatter(np.array(line_2014_02_18_anomaly.Hour_Minute),line_2014_02_18_anomaly.value, marker='+',color = 'c', label='line_2014_02_18_anomaly_flag')


    # ！！！！！
    # if len(line_2014_02_19_anomaly)>0:
    #     plt.scatter(np.array(line_2014_02_19_anomaly.Hour_Minute),line_2014_02_19_anomaly.value, marker='>',color = 'orange', label='line_2014_02_15_anomaly')
    #因为这里需要画图，所以需要想想如果用上了if-else语句，那么需要使用return吗，return的内容是什么



    plt.plot(line_2014_02_23.Hour_Minute, line_2014_02_23.value, color = 'blue', label='line_2014_02_23',alpha = 0.4)
    if len(line_2014_02_23.loc[line_2014_02_23['anomaly']== 1])>0:
        line_2014_02_23_anomaly = line_2014_02_23.loc[line_2014_02_23['anomaly']== 1]
        plt.scatter(np.array(line_2014_02_23_anomaly.Hour_Minute),line_2014_02_23_anomaly.value, marker='+',color = 'blue', label='line_2014_02_23_anomaly')
    #
    plt.plot(line_2014_02_24.Hour_Minute, line_2014_02_24.value, color = 'purple', label='line_2014_02_24',alpha = 0.4)
    if len(line_2014_02_24.loc[line_2014_02_24['anomaly']== 1])>0:
        line_2014_02_24_anomaly = line_2014_02_24.loc[line_2014_02_24['anomaly']== 1]
        plt.scatter(np.array(line_2014_02_24_anomaly.Hour_Minute),line_2014_02_24_anomaly.value, marker='+',color = 'purple', label='line_2014_02_24_anomaly_flag')


    plt.xlabel('Time')
    plt.ylabel('CPU Utilization Value')

    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # plt.legend(loc = 'best')

    plt.xticks(rotation=30)

    plt.figure(figsize=(32,16))

    # x_major_locator=MultipleLocator(20)
    # ax=plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)

    # plt.show()





































