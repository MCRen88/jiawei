
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

from datetime import datetime,timedelta
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

# def calculate_features(data, window, with_label=True):
def calculate_features(data, window):

    """
    Caculate time features.

    :param data: the time series to detect of
    :param window: the length of window
    """

    features = []
    label = []

    sliding_arrays = sliding_window(data.value, window_len=window)

    for ith, arr in enumerate(sliding_arrays):
        tmp = feature_service.extract_features(arr, window)
        features.append(tmp)
    # if with_label:
    # label = data.anomaly.values[window + 7 * 140 - 1:]
    label = data.anomaly[window + 7 * DAY_PNT - 1:]
    label = pd.DataFrame(label)

    # else:
    #     label = None
    return [features, label]


def Precision_Recall_Curve(y_train, x_train,y_test,x_test):
    """
    Obtain values of Precision and Recall of the prediction;
    Draw Precision_Recall_Curve

    :param y_true: True binary labels.
                If labels are not either {-1, 1} or {0, 1}, then pos_label should be explicitly given.
    :param y_pred_proba: Estimated probabilities or decision function.
    :param pos_label: The label of the positive class.
    """


    precision_train, recall_train, thresholds_train = precision_recall_curve(y_train, anomaly_score_train, pos_label=1)
    precision_test, recall_test, thresholds_test = precision_recall_curve(y_test, anomaly_score_test, pos_label=1)


# def Precision_Recall_assessed_value(precision, recall,draw_type)
    # P-R图
    plt.clf()
    # label_name = 'Precision Recall Curve_{}'.format(draw_type)
    plt.plot(recall_train, precision_train, label= 'Training Data')
    plt.plot(recall_test, precision_test, label= 'Test Data')

    plt.title("Precision Recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc = "best")
    plt.show()
    return precision_train, recall_train, thresholds_train,precision_test, recall_test, thresholds_test

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

def moving_feature(statistic_methods):
    # name = 'moving_{}'.format(statistic_methods)
    array_name = []
    for w in range(0,len(total_dataset)):
        temp = statistic_methods(total_dataset.value[0:w+1])
        array_name.append(temp)
    array_name = pd.DataFrame(array_name)

    return array_name

def millisec_to_str(sec):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(sec)))

def day_based_changed(target_date,tatget_color):
    line_name = 'Line_{}'.format(target_date)
    line = total_dataset_Date_index.loc[target_date] ##12。24为基准
    plt.plot(line.Hour_Minute, line.value, color = tatget_color,  label=line_name,alpha = 0.4)
    if len(line.loc[line['anomaly']== 1])>0:
        line_anomaly = line.loc[line['anomaly']== 1]
        plt.scatter(np.array(line_anomaly.Hour_Minute),line_anomaly.value, marker='+',color = tatget_color, label='line_2014_02_17_anomaly_flag')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.xticks(rotation=30)





def day_based_changed_proba_predict(target_date1,target_date2, target_date3):
    fig = plt.figure()
    ax = fig.add_subplot(111)
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
        ax.scatter(np.array(line_anomaly.Hour_Minute),line_anomaly.value, marker='+',color = 'red', label='line_2014_02_17_anomaly_flag')
    ax2 = ax.twinx()
    lns3 = ax2.plot(line1.Hour_Minute, line1.y_proba_pred_total, label = 'Anomaly Score')
    # lns = lns1+lns2+lns3
    # labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # ax.legend(bbox_to_anchor=(1.0, 0.05), loc=3, borderaxespad=0)
    ax.legend(loc = 'best')
    ax2.legend(loc = 'best')
    plt.xticks(rotation=30)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Value changed")
    ax2.set_ylabel("Probability Anomaly Score")
    ax2.set_ylim(0, 1)
    plt.xticks(rotation = 30)



##########################################--- main ----#####################################################################
##########################################--- main ----#####################################################################

if __name__ == "__main__":
    window = 23
    DAY_PNT = 23
    total_dataset = pd.read_csv('data/ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/A4Benchmark-TS34.csv')


##################数据清洗########################
    ##--1。时间处理格式，顺序--##
    total_dataset['timestamp'] = total_dataset['timestamp'].map(millisec_to_str)
    total_dataset['timestamp'] = total_dataset['timestamp'].map(pd.to_datetime)
    total_dataset = total_dataset.sort_values(by = ['timestamp'],ascending= True)

    ##--2。判断确实值并对缺失数据进行填充，目标只针对timestamp,value,anomaly--##--
    if total_dataset.timestamp.isnull().any() == True:
        total_dataset.dropna(subset=['timestamp'])
    if total_dataset.anomaly.isnull().any() == True:
        total_dataset.dropna(subset=['anomaly'])
    if total_dataset.value.isnull().any() == True:
        total_dataset.fillna(total_dataset.value.mean())


    # #     #数据切分
    training_data, test_data = train_test_split(total_dataset, test_size = 0.3, shuffle=False)
    x_train, y_train = calculate_features(training_data, window)
    x_test, y_test = calculate_features(test_data, window)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train=pd.DataFrame(x_train)
    y_train= pd.DataFrame(y_train)


    ##数据建模
    # 将数据建模放在数据处理部分是因为：数据建模后，得到异常预测值，该预测值会被记录到整个数据集中；；；
    clf = GradientBoostingClassifier(n_estimators=5)
    clf.fit(x_train, y_train)
    clf.fit(x_test, y_test)
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    y_pred_train= clf.predict(x_train)
    y_pred_test = clf.predict(x_test)
    anomaly_score_train = clf.predict_proba(x_train)[:, 1]  # y_train 取值为1（异常）的概率
    anomaly_score_test = clf.predict_proba(x_test)[:, 1]  # y_train 取值为1（异常）的概率


    #数据拆分和合成新列
        #时间数据拆分--根据timestamp添加2个新列--Date和Hour_Minute
    time_extraction_df = pd.DataFrame(total_dataset.timestamp)
    time_extraction_df.rename(columns={'timestamp':'Hour_Minute'},inplace=True)
    date_extraction_df = pd.DataFrame(total_dataset.timestamp)
    date_extraction_df.rename(columns={'timestamp':'Date'},inplace=True)
    time_extraction_df['Hour_Minute']= pd.to_datetime(time_extraction_df['Hour_Minute'], format='%Y-%m-%d %H:%M:%S')
    time_extraction_df['Hour_Minute'] = time_extraction_df['Hour_Minute'].dt.time
    date_extraction_df['Date']= pd.to_datetime(date_extraction_df['Date'], format='%Y-%m-%d %H:%M:%S')
    date_extraction_df['Date'] = date_extraction_df['Date'].dt.date
    total_dataset = pd.concat([total_dataset,date_extraction_df,time_extraction_df],axis= 1)
    total_dataset['Date']=pd.to_datetime(total_dataset['Date'])
    #########total_dataset中含有两新列----Date和Hour_Minute


    #3.a 加入预测值
    #y_pre_total 是total dataset中被预测为anomaly的值
    # y_proba_pred_total是预测概率值
    # -- insert the prediction value of training data then test data
    df1 = pd.DataFrame(columns = ['y_pred_total'])
    df2 = pd.DataFrame(columns = ['y_proba_pred_total'])
    total_dataset = pd.concat([total_dataset,df1,df2])
    anomaly_dataset = total_dataset.loc[total_dataset['anomaly']== 1] ##anomaly_dataset数据集提取了total_dataset中是被标记为1的数据

    # 创建预测值的dataframe
    y_pred_train_df = pd.DataFrame(y_pred_train, columns= {'y_pred_train'})
    y_pred_test_df = pd.DataFrame(y_pred_test,columns= {'y_pred_test'})
    y_predict_proba_train_df =  pd.DataFrame(anomaly_score_train, columns= {'y_proba_pred_train'})
    y_predict_proba_test_df =  pd.DataFrame(anomaly_score_test, columns= {'y_proba_pred_test'})

    # #窗口长度
    training_data_lenth = int(0.7 * len(total_dataset)) #整个测试集的大小
    window_test_range = window + 7* DAY_PNT #对于每个测试集所抽取 e d
    #加入predic和prob_predict值
    # 加入training
    for i in range(window_test_range,training_data_lenth):
        total_dataset.ix[i,'y_pred_total'] = y_pred_train_df.ix[i-window_test_range,'y_pred_train']
        total_dataset.ix[i,'y_proba_pred_total'] = y_predict_proba_train_df.ix[i-window_test_range,'y_proba_pred_train']
    # for i in range((training_data_lenth + window_test_range), len(total_dataset)):
    #     total_dataset.ix[i,'y_pred_total'] = y_pred_test_df.ix[i-window_test_range,'y_pred_test']
    #     total_dataset.ix[i,'y_proba_pred_train_total'] = y_predict_proba_test_df.ix[i-window_test_range,'y_proba_pred_test']



    total_dataset_Date_index = total_dataset
    total_dataset_Date_index = total_dataset_Date_index.set_index('Date')
    #3.原始数据的value走向和被标记为数据异常点的观察
    #3.a 加入新列，用于放预测是否诗异常的值 -- 没有发现有明显标记错误的点
    anomaly_dataset = total_dataset.loc[total_dataset['anomaly']== 1]
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)), total_dataset.value, color='blue', label='Value of CPU Utilization')
    plt.scatter(pd.to_datetime(np.array(anomaly_dataset.timestamp)), anomaly_dataset.value, color = 'red', label='Anomaly Value of CPU Utilization - Flag')
    plt.legend(loc='best')
    plt.xticks(rotation=30)
    plt.show()

    #以12。22为基准，画前一天，后一天，前后1，2周的曲线
    day_based_changed('2014-12-22','red')
    day_based_changed('2014-12-23','orange')
    day_based_changed('2014-12-21','yellow')
    day_based_changed('2014-12-15','green')
    day_based_changed('2014-12-08','blue')
    day_based_changed('2014-12-29','purple')
    day_based_changed('2015-01-05','black')
    plt.xlabel('Time')
    plt.ylabel('Production Traffic')
    plt.show()

###########################################----提取特征 start---#################################################
############################################----提取特征 start---#################################################
    moving_time_series_mean = moving_feature(time_series_mean)
    moving_time_series_variance = moving_feature(time_series_variance)
    moving_time_series_standard_deviation = moving_feature(time_series_standard_deviation)
    moving_time_series_skewness = moving_feature(time_series_skewness)
    moving_time_series_kurtosis = moving_feature(time_series_kurtosis)
    moving_time_series_median = moving_feature(time_series_median)
    moving_time_series_abs_energy = moving_feature(time_series_abs_energy)
    moving_time_series_absolute_sum_of_changes = moving_feature(time_series_absolute_sum_of_changes)
    moving_time_series_variance_larger_than_std = moving_feature(time_series_variance_larger_than_std)
    moving_time_series_sum_values = moving_feature(time_series_sum_values)
    moving_time_series_maximum = moving_feature(time_series_maximum)
    moving_time_series_minimum = moving_feature(time_series_minimum)
    moving_time_series_sum_values = moving_feature(time_series_sum_values)
    moving_time_series_range = moving_feature(time_series_range)



    total_dataset['moving_time_series_mean'] = moving_time_series_mean
    total_dataset['moving_time_series_variance'] = moving_time_series_variance
    total_dataset['moving_time_series_standard_deviation'] = moving_time_series_standard_deviation
    total_dataset['moving_time_series_skewness'] = moving_time_series_skewness
    total_dataset['moving_time_series_kurtosis'] = moving_time_series_kurtosis
    total_dataset['moving_time_series_median'] = moving_time_series_median
    total_dataset['moving_time_series_abs_energy'] = moving_time_series_abs_energy
    total_dataset['moving_time_series_absolute_sum_of_changes'] = moving_time_series_absolute_sum_of_changes
    total_dataset['moving_time_series_variance_larger_than_std'] = moving_time_series_variance_larger_than_std
    total_dataset['moving_time_series_sum_values'] = moving_time_series_sum_values
    total_dataset['moving_time_series_maximum'] = moving_time_series_maximum
    total_dataset['moving_time_series_minimum'] = moving_time_series_minimum
    total_dataset['moving_time_series_range'] = moving_time_series_range
    total_dataset['moving_time_series_sum_values'] = moving_time_series_sum_values


    # moving_time_series_sum_values = moving_feature(time_series_maximum)
    #
    # #########!!!!在画图之前，应该有if-else语句，能够判断大于3-5倍？的就不画这个图 #####################
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_mean, color='red', label='moving time series mean')
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_standard_deviation, color='orange', label='moving time series standard deviation')
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_skewness, color='#FFD400', label='moving time series skewness')
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_kurtosis, color='#11FF0A', label='moving time series kurtosis')
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_median, color='blue', label='moving time series median')
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.value, color='black', label='The Production Traffic of Yahoo',alpha = 0.3)
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_maximum, color='purple', label='moving time series maximum')
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_minimum, color='#0080FF', label='moving time series minimum')
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_sum_values, color='#FF5733', label='moving time series sum values')
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_range, color='#FF6A5D', label='moving time series range')
    plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_sum_values, color='pink', label='moving time series sum values')

    total_dataset = total_dataset.set_index("anomaly")
    anomaly_total = total_dataset.loc[1]
    plt.scatter(pd.to_datetime(np.array(anomaly_total.timestamp)),anomaly_total.value, color='black',marker='+', label='Anomaly_Flag')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.xticks(rotation=30)
    plt.show()


###########################################----提取特征 end---#################################################


###########################################----数据评估-start--########################################################


    #和anomaly_score进行对比 --假设对12。22进行预测
    day_based_changed_proba_predict('2014-12-22','2014-12-21','2014-12-15')
    plt.show()

    precision_train, recall_train, thresholds_train,precision_test, recall_test, thresholds_test = Precision_Recall_Curve(y_train,x_train,y_test,x_test) # P_R-Curve
    plt.plot(precision_train,color = 'red',label = 'precision_train')
    plt.plot(recall_train,color = 'blue',label = 'recall_train')
    plt.legend(loc= 'best')
    plt.show()
###########################################----数据评估-end--########################################################



##############################################--end--#############################################################################
##############################################--end--#############################################################################








####将被预测为异常的数据添加到 total_dataset 中



###### 提取出有问异常的数据和被预测为异常的数据
# total_dataset_anomaly = total_dataset.loc[total_dataset['y_pred_total'] == 1]
#
# if len(total_dataset_anomaly) > 0:
#     plt.scatter(pd.to_datetime(np.array(total_dataset_anomaly.timestamp)), total_dataset_anomaly.value, color='black', marker='+',label='Anomaly Value of CPU Utilization - Predicted')
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)), total_dataset.value, color='blue', label='Value of CPU Utilization')
# plt.scatter(pd.to_datetime(np.array(anomaly_dataset.timestamp)), anomaly_dataset.value, color='red', label='Anomaly Value of CPU Utilization - Flag')
# plt.legend(loc = 'best')
# plt.xticks(rotation=30)
# plt.show()
#############



#得到新的数据列，并将新的数据列加入到total_dataset中



# #数据展示

#     # print pd.DataFrame(y_train.loc[y_train == 1])
# print pd.DataFrame(total_dataset.loc[total_dataset['anomaly']== 0])
#
#     # y_pred_train = clf.predict(np.array(training_data.value))
#     # y_pred_test = clf.predict(x_test)


###





    ###### 提取出有问异常的数据和被预测为异常的数据
    # total_dataset_anomaly = total_dataset.loc[total_dataset['y_pred_total'] == 1]
    #
    # if len(total_dataset_anomaly) > 0:
    #     plt.scatter(pd.to_datetime(np.array(total_dataset_anomaly.timestamp)), total_dataset_anomaly.value, color='black', marker='+',label='Anomaly Value of CPU Utilization - Predicted')
    # plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)), total_dataset.value, color='blue', label='Value of CPU Utilization')
    # plt.scatter(pd.to_datetime(np.array(anomaly_dataset.timestamp)), anomaly_dataset.value, color='red', label='Anomaly Value of CPU Utilization - Flag')
    # plt.legend(loc = 'best')
    # plt.xticks(rotation=30)
    # plt.show()
    #############

    # plt.xlabel('Time')
    # plt.ylabel('Production Traffic')
    # plt.show()








# print precision_train,recall_train
#     # print training_data[-100:]
#     # addpred_training_data = training_data
#     # addpred_training_data['y_train_pred'] = pd.DataFrame(y_pred_train)
#     # addpred_training_data = pd.concat([training_data, pd.DataFrame(y_pred_train)])


#
#
#     # pred_anomaly_dataset = training_data.loc[training_data['anomaly']== 1]
#
########################################################数据评估######################################################### #图中需要涉及的内容
# #


#

#
#
    #########################################-----line_以天为周期///开始//-----####################################################################3
    # !!!!!!!!line_以天为周期，抽取4天的CPU使用情况进行绘图，图中每条曲线代表特定日期的CPU使用情况，同色系的散点图代表被标记为异常的点；
    # 【【【【【【【【【【【【【【【【






#     ###】】】】】】】】】】】】】】】】】】】】
#     ##########################################-----line_以天为周期///结束//-----####################################################################3
#
#
#
#     # mean = np.mean(total_dataset.value)
#
#     # plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)), time_series_moving_average(total_dataset.value), color='blue', label='Value of CPU Utilization')
#
#     # plt.show()
#     # print fitting_features.extend(time_series_moving_average(x_list[4]))
##################################################################################################################################################################################################################################################################################







################################################----------for test about the features selection---start---------#########################################################################
################################################----------for test about the features selection---start---------#########################################################################

    # mean_value = df.value
    # mean_value.rename(columns = {'mean_value'})
    # all = pd.concat([df,mean_value],axis=1)



    #只是用来验证df里面有异常数据，通过将数据切片减少用来分析的数据
    #df中因为有异常数据，现在当作假想的total dataset进行验证--
    # ano = df.set_index("anomaly")
    # a = ano.loc[1]


    #####moving_average
    #
    # x= total_dataset.value
    # moving_mean = []
    # for w in range(0,len(total_dataset)):
    #     temp = np.mean(total_dataset.value[0:w+1])
    #     moving_mean.append(temp)
    # moving_mean = pd.DataFrame(moving_mean)
    # print moving_mean
    # print moving_mean
    #
    #

#
#
##############################以下是放弃的特征############################################

# a = moving_feature(time_series_variance_larger_than_std)
# b = moving_feature(time_series_count_above_mean)
# c = moving_feature(time_series_variance_larger_than_std)
# d = moving_feature(time_series_count_above_mean)
# e = moving_feature(time_series_count_below_mean)
# f = moving_feature(time_series_first_location_of_maximum)
# g = moving_feature(time_series_first_location_of_minimum)
# h = moving_feature(time_series_last_location_of_maximum)
# i = moving_feature(time_series_last_location_of_minimum)
# j = moving_feature(time_series_has_duplicate)
# k = moving_feature(time_series_has_duplicate_max)
# l = moving_feature(time_series_has_duplicate_min)
# m = moving_feature(time_series_longest_strike_above_mean)
# n = moving_feature(time_series_longest_strike_below_mean)
# o = moving_feature(time_series_mean_abs_change)
# p = moving_feature(time_series_mean_change)
# q = moving_feature(time_series_percentage_of_reoccurring_datapoints_to_all_datapoints)
# r = moving_feature(time_series_ratio_value_number_to_time_series_length)
# s = moving_feature(time_series_sum_of_reoccurring_data_points)
# t = moving_feature(time_series_sum_of_reoccurring_values)
# u = moving_feature(time_series_longest_strike_below_mean)
# time_series_maximum(x),1
# time_series_minimum(x),1
# time_series_variance_larger_than_std(x),
# time_series_count_above_mean(x),
# time_series_count_below_mean(x),
# time_series_first_location_of_maximum(x),
# time_series_first_location_of_minimum(x),
# time_series_last_location_of_maximum(x),
# time_series_last_location_of_minimum(x),
# int(time_series_has_duplicate(x)),
# int(time_series_has_duplicate_max(x)),
# int(time_series_has_duplicate_min(x)),
# time_series_longest_strike_above_mean(x),
# time_series_longest_strike_below_mean(x),
# time_series_mean_abs_change(x),
# time_series_mean_change(x),
# time_series_percentage_of_reoccurring_datapoints_to_all_datapoints(x),
# time_series_ratio_value_number_to_time_series_length(x),
# time_series_sum_of_reoccurring_data_points(x),
# time_series_sum_of_reoccurring_values(x),
# time_series_sum_values(x),
# time_series_range(x)


# total_dataset['a'] = a
# total_dataset['b'] = b
# total_dataset['c'] = c
# total_dataset['d'] = d
# total_dataset['e'] = e
# total_dataset['f'] = f
# total_dataset['g'] = g
# total_dataset['h'] = h
# total_dataset['i'] = i
# total_dataset['j'] = j
# total_dataset['k'] = k
# total_dataset['l'] = l
# total_dataset['m'] = m
# total_dataset['n'] = n
# total_dataset['o'] = o
# total_dataset['p'] = p
# total_dataset['q'] = q
# total_dataset['r'] = r
#
# total_dataset['s'] = s
# total_dataset['t'] = t
# total_dataset['u'] = u
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.s, color='purple', label='moving_time_series_minimum')
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.t, color='pink', label='moving_time_series_maximum')
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_minimum, color='orange', label='moving_time_series_minimum')
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_maximum, color='red', label='moving_time_series_maximum')
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_minimum, color='orange', label='moving_time_series_minimum')
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_maximum, color='red', label='moving_time_series_maximum')
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_minimum, color='orange', label='moving_time_series_minimum')
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_maximum, color='red', label='moving_time_series_maximum')
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_minimum, color='orange', label='moving_time_series_minimum')
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_variance, color='orange', label='moving time series variance')
###!!!!这两个在statisti f里面去掉
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_absolute_sum_of_changes, color='black', label='moving time series sum_of_changes')
# plt.plot(pd.to_datetime(np.array(total_dataset.timestamp)),total_dataset.moving_time_series_abs_energy, color='pink', label='moving time series abs_energy')


#moving_time_series_sum_values
#
# total_dataset = total_dataset.set_index("anomaly")
# print total_dataset
# anomaly_total = total_dataset.loc(1)



#####################################













    # temp_list = []
    # for w in range(0, len(total_dataset)):
    #     temp = np.mean(total_dataset[0:w+1])
    #     temp_list.append(temp)
    # temp_list= pd.DataFrame(temp_list)
    # print temp_list.head(10)
    #     list(np.array(total_dataset) - x[-1])










################################################----------for test about the features selection---END---------#########################################################################
################################################----------for test about the features selection---END---------#########################################################################























    # total_dataset['Date'] = time.strftime("%Y-%M-%D", time.gmtime(float(np.str(total_dataset['Date']))))
    # print time.strftime("%Y-%m-%d", time.gmtime(total_dataset['Date']))

    # print add_all_character['Date']
# time.gmtime(float(np.str(add_all_character['Date'])))


# dataset_daytime_extraction.Hour_Minute = \





    # def plt.plot(dataset_name, color , marker ,label):







# ax=plt.gca()
# date_format=mpl.dates.DateFormatter('%Y-%m-%d')%Y-%m-%d %H:%M:%S#设定显示的格式形式
# ax.xaxis.set_major_formatter(date_format)#设定x轴主要格式
# ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(30))#设定坐标轴的显示的刻度间隔
# fig.autofmt_xdate()#防止x轴上的数据重叠，自动调整。
# ---------------------
# 版权声明：本文为CSDN博主「lishangyin88」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/lishangyin88/article/details/80219433

    # total_dataset_top1000 = total_dataset.head(1000)
    # total_dataset.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(total_dataset.timestamp))


#数据显示
    # total_dataset.timestamp1 = total_dataset.timestamp.astype
    # total_dataset.timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(total_dataset.timestamp))


    # total_dataset_top1000.timestamp = total_dataset_top1000.timestamp.map(millisec_to_str)
    # df = total_dataset_top1000
    # df["value"] = df["point"]
    # df["timestamp"] = df["timestamp"].map(millisec_to_str)
    # df["timestamp"] = df["timestamp"].map(pd.to_datetime)
    # df = df.set_index ("timestamp")
    # plt.figure(figsize=(16,7))
    # df["value"].plot(lable= "today",alpha=0.8)
    # plt.legend()


    # datetime_timestamp_total = pd.to_datetime (total_dataset.timestamp)
    # total_dataset_timestamp_toseries = pd.Series(total_dataset.timestamp)
    # total_dataset_timetoarray = np.array(total_dataset.timestamp)

    # plt.scatter(total_dataset_timetoarray,total_dataset.value, color='blue', label='value of the total dataset')
    # anomaly_dataset_timetodatetime = pd.to_datetime(anomaly_dataset.timestamp)
    # plt.scatter(anomaly_dataset_timetodatetime, anomaly_dataset.value, color='red', label='value of the total dataset')
    # x=  pd.to_datetime(np.array(total_dataset.timestamp))
    # y = total_dataset.value.values
    # plt.scatter(pd.to_datetime(np.array(total_dataset.timestamp)), total_dataset.value, color='blue', label='value of the total dataset')

    # ax.axis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))  #设置时间显示格式
    #

    # ax = plt.gca()
    # ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))  # 横坐标标签显示的日期格式



    # print total_dataset.head(10)





    # plt.scatter(total_dataset.anomaly, color='red', label='flag')








# 数据处理：
    #
    # 观察原数据走向，观察数据特征分布--需要在同一张图中
    # y_pred = clf.predict(x)
    # plt.plot(y, color='blue', label='true data')
    # plt.plot(y_pred, color='red',label='predict data')
    # plt.legend(loc='best')
    # plt.title('Comparison of true and predict curve')
    # plt.show(block=False)


    # 选择适合进行检测的特征；






    #
    # plt.plot(y_pred_train, color='blue', label='value of the total dataset')
    # plt.plot(y_train, color='red', label='value of the total dataset')
    # # plot_pr.xlim([0.0, 1.0])
    # # plot_pr.ylim([0.0, 1.0])
    # plt.legend(loc='best')
    # plt.show()

    # # ######  P-R 图，不能delete   ###############
    # true_predict_curve(y_train, x_train)
    # true_predict_curve(y_test, x_test) #true data完全缺失？？？




    # print total_dataset.value.loc[[2]],'\n\n\n',total_dataset.loc[[2]]
    # print '\n\n\n',pd.DataFrame(y_pred_test).loc[2]

















    #画出x_train的曲线图
    # plt.plot(test_data.timestamp, test_data.value, label='test_data')
    # plt.xlabel('timestamp')
    # plt.ylabel('value')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.show()

    # test_data.value.plot(kind='kde')































#混淆矩阵
    # CM = confusion_matrix(y_test, y_pred)
    # print CM
    #后续加入不同的评估函数


    # clf.fit(x_train, y_train)
    # print clf.predict(x_train)
    # print f1_score(y_train, clf.predict(x_train))
    # anomaly_score = [int(item> threshold) for  item in anomaly_score]
    # CM = confusion_matrix(y_train, clf.predict(x_train))
    # y_true = CM[1,1]
    # print y_true


    #需要加入precision和recall的值吗
    #precision，recall
    #



    # plot_pr(0.5, precision, recall, "pos")


    #roc图
    # 原文链接：https://blog.csdn.net/xyz1584172808/article/details/81839230

    # Compute ROC curve and ROC area for each class
    # fpr_roc,tpr_roc,threshold_roc = roc_curve(y_test, y_pred) ###计算真正率和假正率




    # roc_auc = auc(fpr_roc,tpr_roc) ###计算auc的值
    #
    # plt.figure()
    # lw = 2
    # plt.figure(figsize=(10,10))
    # plt.plot(fpr_roc, tpr_roc, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
    # ---------------------


    # fpr_roc,tpr_roc,threshold_roc = roc_curve(y_train,anomaly_score, pos_label='1', sample_weight=None, drop_intermediate=True)
    # roc.show()
    # #因为还没有多个指标；；后续添加
    # classification_report(y_true, y_pred, target_names=target_names)

    # confusion_matrix(test['Survived'], predictions)



    #先跳过；有空研究下载
    # plot_precision_recall_curve(clf, recall, precision)
    # plt.show()



