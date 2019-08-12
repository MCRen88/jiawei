
# coding: utf-8

import numpy as np
import csv
import pandas as pd
import re
import matplotlib.pylab as plt
import tsfresh.feature_extraction.feature_calculators as ts_feature_calculators
from scipy.cluster.vq import whiten
from sklearn.metrics import f1_score,precision_score
from sklearn.model_selection import train_test_split
from numpy import column_stack
# from imblearn.over_sampling import SMOTE #数据不均衡


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

#import detect
from time_series_detector.detect import *

#import algorithm
from time_series_detector.algorithm.ewma import *
from time_series_detector.algorithm.xgboosting import *
from time_series_detector.algorithm.isolation_forest import *
from time_series_detector.algorithm.gbdt import *

from time_series_detector.common.tsd_common import DEFAULT_WINDOW, split_time_series



#数据为realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv的数据；
#数据是从2.14.2.14-2.28间每隔5分钟的数据；
#已经完成异常标记；1表示异常，0表示正常

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


#
#原始record
# def calculate_features(data, window, with_label=True):
#     """
#     Caculate time features.
#     :param data: the time series to detect of
#     :param window: the length of window
#     """
#     features = []
#     sliding_arrays = sliding_window(data.value, window_len=window)
#     for ith, arr in enumerate(sliding_arrays):
#         tmp = feature_service.extract_features(arr, window)
#         features.append(tmp)
#     if with_label:
#         label = data.anomaly.values[window + 7 * 100-1:]
#     else:
#         label = None
#     return [features, label]

##########################test_by shihuan######################################
# if __name__ == "__main__":
#     window = 60
#     test_data = pd.read_csv('data/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_6.csv')
#     X_train, y_train = calculate_features(test_data, window)
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
#     clf = GradientBoostingClassifier(n_estimators=5)
#     clf.fit(X_train, y_train)
#     print clf.predict(X_train)


###################for showing the problem ##################
if __name__ == "__main__":
    window = 60
    train_ds = pd.read_csv('data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv')
    train_data, test_data = train_test_split(train_ds, test_size=0.3, shuffle=False)

    # X = train_ds
    # train_size = int(len(X) * 0.66)
    # train_data, test_data = X[1:train_size], X[train_size:]
    # x_train, x_test = train_data[:,0], train_data[:,1]
    # y_train, y_test = test_data[:,0], test_data[:,1]
    # train_data, test_data = train_ds[:80,:], train_ds[80:,:]
    # print "train_data\n",train_data,test_data
    # # print "total_data_\n",train_ds.head(2)
    # print "train_data.value\n",train_data.value.head(2)
    # print "train_data.flag\n",train_data.is_anomaly.head(2)
    # print "type",type(train_data) == type(train_ds)

    # print type(train_ds)
    # train_data = train_ds[0:train_size]
    # over_samples = SMOTE(random_state=0)
    # over_samples_x, over_samples_y = over_samples.fit_sample(x_train, y_train)
    # over_samples_x, over_samples_y = over_samples.fit_sample(x_train.values,y_train.values.ravel())


    x_train, y_train = calculate_features(train_data, window)
    x_test, y_test = calculate_features(test_data, window)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print "x_train array", x_train
    print "y_train array", y_train
    clf = GradientBoostingClassifier(n_estimators=5)



    # clf.fit(x_train, y_train)
    # print clf.predict(x_train)
    #
    # print f1_score(y_train, clf.predict(x_train))
    # print f1_score(y_test, clf.predict(x_test))
    # print precision_score(y_train, clf.predict(x_train))


    clf.fit(x_train, y_train)
    print clf.predict(x_train)
    print  f1_score(y_train, clf.predict(x_train))




#
#
# # #####################    iiiiiii for learning  ###########################
# if __name__ == "__main__":
#     window = 60
#     # test_data = pd.read_csv('data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv')
#     train_ds = pd.read_csv('data/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_6.csv')
#
#
#     # split into train and test sets
#     X = train_ds.values
#     train_size = int(len(X) * 0.66)
#     train_data, test_data = X[1:train_size], X[train_size:]
#     x_train, x_test = train_data[:,0], train_data[:,1]
#     y_train, y_test = test_data[:,0], test_data[:,1]
#
#
#     # test = pd.read_csv('data/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_8.csv')
#     # X = train_ds.value
#     # y = train_ds.is_anomaly
#     x_train, y_train, x_test, y_test = train_test_split(train_ds.value, train_ds.is_anomaly, test_size=0.3)
#     print x_train, y_train, x_test, y_test

    # train_data, test_data = train_test_split(train_ds, test_size=0.3, shuffle=False)
    # # print train_data.head(10)
    # print "traindata_head3\n",train_data.head(3)
    # print "totaldata_head3\n",train_ds.head(3)


    # x_train = train_data.value
    # print "xxxhead3\n",x_train.head(3)
    #
    #
    # y_train = train_data.is_anomaly
    # print  "yyyhead3\n",y_train.head(3)


    # train_size = int(len(train_ds) * 0.7)
    # train_data, test_data = train_ds[0:train_size], train_ds[train_size:len(train_ds)]
    # print train.value,"########\n",train.is_anomaly,"########\n",train_data, "########\n",test_data

    # train_data = np.column_stack({x_train,y_train})
    # train_data = pd.DataFrame(x_train, y_train)
    # train_data.name = ('value', 'is_anomaly')

    # print train_ds, train_data


    # train_data =train_data.toArray()
    # print train_data
    # train_data = pd.DataFrame((x_train))

    # x_train = pd.DataFrame(x_train)
    # y_train = pd.DataFrame(y_train)
    # train_data = pd.concat([x_train, y_train],axis=1，index=1)
    # x_train = train_data.value
    # y_train = train_data.is_anomaly
    # a=train_data.timestamp




    # train_data=pd.DataFrame(columns={"value":"","is_anomaly":""},index=[0])

    # print train_data
    # train_data = pd.concat([pd.DataFrame(x_train,columns=['m']),pd.DataFrame(y_train,columns=['m'])],axis=1)

    # print train_data.head(10),"\n\n\n\n",train_ds.head(10)
    # train_data = np.array(train_data)
    # print type(train_data)
    # print type(train_ds)
    #
    # x_train, y_train,= calculate_features(train_data, window)
    # # x_train = np.array(x_train)
    # # y_train = np.array(y_train)
    # print "x_train array", x_train
    # print "y_train array", y_train
    # print x_train

    # print "YYYYYYYYYYY\n",y_train
    # # # X,y = calculate_features(train_ds, window)
    # X_train, y_train = calculate_features(train_data, window)



    # print x_train.dtype
    # print "VVVVVVVvvvvvVVVVVVVV\n",x_train.head(3)
    # print "AAAAAAAAAAAAAAAAAAAA\n",y_train
    #
    # x_train = np.array(x_train)
    # print "VVVVVVVvvvvvVVVVVVVV\n",x_train
    # y_train = np.array(y_train)
    # print "AAAAAAAAAAAAAAAAAAAA\n",y_train
    # # # print train_data,"\n",test_data
    #
    # print X_train,"\n\n\n\n\n\n",y_train, "\n\n\n\n",train_data

    # print train_data,"\n",train_ds,X,"\n",X_train



    # X_test = np.array(X_test)
    # y_test = np.array(y_test)
    #
    # print train_ds
    # clf = GradientBoostingClassifier(n_estimators=5)
    # clf.fit(x_train, y_train)
    # print clf.predict(x_train)
    # print clf.predict(X_test)

    # print f1_score(y_test, clf.predict(X_test))
    # print precision_score(y_test, clf.predict(X_test))

