
# coding: utf-8

import numpy as np
import csv
import pandas as pd
import re
import matplotlib.pylab as plt
import tsfresh.feature_extraction.feature_calculators as ts_feature_calculators
from scipy.cluster.vq import whiten

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
# test_data = pd.read_csv('data/realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv')
# ts = pd.Series(test_data_data['timestamp'].values, index=test_data['value'])




# train=test_data.head(1500)
# test = test_data.tail(500)


#标准化value值

#split value值并将split的值标准化
# split_ts = split_time_series(list(test_data.value))
# normalized_split_value = tsd_common.normalize_time_series(split_ts)



# features = []
# for index in test_data:
#     if is_standard_time_series(index["data"], 6):
#         print("y")
#     else:
#         temp = []
#         temp.append(feature_service.extract_features((test_data.value), 6)
#         temp.append(index)
#         features.append(temp)
#     print features

# print("temp:")
# print (temp)
# # print("#####################")
# print("index:")
# print(index)
# # print(index)
# print("features")




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
        label = data.anomaly.values[window + 7 * DAY_PNT-1:]
    else:
        label = None
    return [features, label]


##########################test_by shihuan######################################
if __name__ == "__main__":
    window = 60
    test_data = pd.read_csv('data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv')
    X_train, y_train = calculate_features(test_data, window)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    clf = GradientBoostingClassifier(n_estimators=5)
    clf.fit(X_train, y_train)
    print clf.predict(X_train)




# 特征提取，并将提取的feature和对应的标签放入features中；
# features = []
# for index in test_data.iterrows():
#     temp = []
#     temp.append(feature_service.extract_features(test_data.value, 6))  #从value中提取的特征
#     temp.append(list(test_data.anomaly))#从标签中提取对应的
#     d = []
#     d.append(feature_service.extract_features(test_data.value, 6))  # 从value中提取的特征
#     d.append(index[2])  # 从标签中提取对应的
#     features.append(temp)
# print index
# print ("a")
# print("-----------")
#
# print(temp)
# print(d)

#
# print("-----------")
# print (features)
#
# temp = []
# temp.append(feature_service.extract_features(test_data.value, 6))
# temp.append(test_data.anomaly)
# print ("a")
# # print (index)
# print (features)
# #




# print ("temp finished")
#
# print(test_data.head(3))



#-------------

#
# X_train = []
# y_train = []
# features = temp
# # if features:
# #     return TSD_LACK_SAMPLE
# for index in features:
#     X_train.append(index[0])
#     y_train.append(index[1])
# X_train = np.array(X_train).reshape(1, -1)
# y_train = np.array(y_train)
# # try:
# grd = GradientBoostingClassifier(n_estimators=300, max_depth=10,
#                                      learning_rate=0.05)
# model = grd.fit(X_train, y_train)
#     # model_name = MODEL_PATH + task_id + "_model"
#     # joblib.dump(grd, model_name)
# # except Exception as ex:
# #     TSD_TRAIN_ERR, str(ex)
#
# # return TSD_OP_SUCCESS, ""
#
#
# print ("train finished")
# #----------------------------------------------------------------------------------------------------------------------
#
# # if is_standard_time_series(normalized_split_value):
# ts_features = feature_service.extract_features(test_data.value, 6)
# ts_features = np.array([ts_features])
# # load_model = pickle.load(open(model_name, "rb"))
# load_model = model
# gbdt_ret = load_model.predict_proba(ts_features)[:, 1]
# if gbdt_ret[0] < self.threshold:
#     value = 0
# else:
#     value = 1
# print("[value, gbdt_ret[0]]")
# # else:
# #      print ("[0, 0]")
#
# print ("test finished")
#
#
#
#
#
#
# #-----------------------------------------------
#
#
#
#
