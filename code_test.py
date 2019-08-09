
# coding: utf-8

# In[1]:


import numpy as np
import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pylab as plt
import tsfresh.feature_extraction.feature_calculators as ts_feature_calculators
import time_series_detector.feature.fitting_features
import tsfresh.feature_extraction.feature_calculators as ts_feature_calculators

from matplotlib.pylab import rcParams

#import feature
from time_series_detector.feature.statistical_features import *
from time_series_detector.feature.classification_features import *
from time_series_detector.feature.feature_service import *
from time_series_detector.feature.fitting_features import *
from time_series_detector.feature.statistical_features import time_series_mean, time_series_variance, time_series_standard_deviation, time_series_median


#import detect
from time_series_detector.detect import *

#import algorithm
from time_series_detector.algorithm.ewma import *
from time_series_detector.algorithm.xgboosting import *
from time_series_detector.algorithm.isolation_forest import *
from time_series_detector.algorithm.gbdt import *


from time_series_detector.common.tsd_common import DEFAULT_WINDOW, split_time_series

DEFAULT_WINDOW=6

# ! pip install --upgrade pip


# import sklearn
# sklearn.__version__

#数据为realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv的数据；
#数据是从2.14.2.14-2.28间每隔5分钟的数据；
#已经完成异常标记；1表示异常，0表示正常
test_data = pd.read_csv('time_series_detector_testdata/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv')

test_data.head(3)


# In[7]:


test_data.dtypes


# In[9]:


p = get_fitting_features(test_data.value)


# In[11]:


time_series_moving_average(test_data.value[4])


# In[13]:


for w in range(1, min(50, 6), 5):
         temp = np.mean(test_data.value[-w:])
         temp_list.append(temp)
        return list(np.array(temp_list) - x[-1])


# In[6]:


test_data.plot()


# In[38]:


#标准化数据
test_data.nmvalue= get_fitting_features([test_data.value,test_data.value,test_data.value,test_data.value,list(test_data.value)])


# In[34]:


test_data.value.iloc[-1]


# In[41]:


#statistical_features
#td_categories为描述性统计的计算值
# td_categories = get_statistical_features(test_data.value)


# In[10]:


# X=test_data.value
# window=7
# x_train = list(range(0, 2 * window + 1)) + list(range(0, 2 * window + 1)) + list(range(0, window + 1))
# sample_features = zip(x_train, X)
# sample_features


# In[11]:


# len(x_train)


# In[40]:


#获取曲线的特征变量
classification_feature = get_classification_features(test_data.value)


# In[13]:


test_data.nmvalue.plot()


# In[20]:


from time_series_detector.algorithm.gbdt import *


# In[16]:


#使用gbdb进行预测
gbdt = Gbdt()
gbdt.gbdt_train(test_data.value,1,6)


# In[33]:





# In[29]:


gbdt.gbdt_train(td_classification_feature,1,window=DEFAULT_WINDOW)


# In[30]:


# isolation_forest.predict(test_data.values)


# In[31]:


td_classification_feature


# In[28]:


# class IForest(object):
#     """
#     The IsolationForest 'isolates' observations by randomly selecting a feature and then
#     randomly selecting a split value between the maximum and minimum values of the selected feature.
#     https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
#     """

#     def __init__(self,
#                  n_estimators=3,
#                  max_samples="auto",
#                  contamination=0.15,
#                  max_feature=1.,
#                  bootstrap=False,
#                  n_jobs=1,
#                  random_state=None,
#                  verbose=0):
#         """
#         :param n_estimators: The number of base estimators in the ensemble.
#         :param max_samples: The number of samples to draw from X to train each base estimator.
#         :param coefficient: The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function.
#         :param max_features: The number of features to draw from X to train each base estimator.
#         :param bootstrap: If True, individual trees are fit on random subsets of the training data sampled with replacement. If False, sampling without replacement is performed.
#         :param random_state: If int, random_state is the seed used by the random number generator;
#                               If RandomState instance, random_state is the random number generator;
#                               If None, the random number generator is the RandomState instance used  by `np.random`.
#         :param verbose: Controls the verbosity of the tree building process.
#         """
#         self.n_estimators = n_estimators
#         self.max_samples = max_samples
#         self.contamination = contamination
#         self.max_feature = max_feature
#         self.bootstrap = bootstrap
#         self.n_jobs = n_jobs
#         self.random_state = random_state
#         self.verbose = verbose

#     def predict(self, X, window=DEFAULT_WINDOW):
#         """
#         Predict if a particular sample is an outlier or not.
#         :param X: the time series to detect of
#         :param type X: pandas.Series
#         :param window: the length of window
#         :param type window: int
#         :return: 1 denotes normal, 0 denotes abnormal.
#         """
#         x_train = list(range(0, 2 * window + 1)) + list(range(0, 2 * window + 1)) + list(range(0, window + 1))
#         sample_features = zip(x_train, X)
#         clf = IsolationForest(self.n_estimators, self.max_samples, self.contamination, self.max_feature, self.bootstrap, self.n_jobs, self.random_state, self.verbose)
#         clf.fit(sample_features)
#         predict_res = clf.predict(sample_features)
# #         clf.fit(test_data.value)
#         test_data.value = check_array(test_data.value, accept_sparse=['csc'])

        
        
        
# #         if predict_res[-1] == -1:
# #             return 0
# #         return 1

