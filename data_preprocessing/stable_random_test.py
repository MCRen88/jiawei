#!/usr/bin/env python
# -*- coding=utf-8 -*-


import pandas as pd
from statsmodels.stats.diagnostic import unitroot_adf
from statsmodels.tsa.stattools import kpss
from statsmodels.stats.diagnostic import acorr_ljungbox


##平稳性测试
def adf_(timeseries): # adf_ 检验平稳性
    """

    :param timeseries: time series that aims to analyse
    :return: the values of the adfuller test and critical test, in order to determine whether the time series is stable or not
    """
    adf_test = unitroot_adf(timeseries)
    adf_test_value = adf_test[0]
    adfuller_value = pd.DataFrame({key:value for key,value in adf_test[4].items()},index = [0])
    adfuller_value = pd.DataFrame(adfuller_value)
    adfuller_critical_value = adfuller_value['10%'][0]
    return adf_test_value, adfuller_critical_value


def kpss_(timeseries): #kpss检验平稳性
    """

    :param timeseries: time series that aims to analyse
    :return: the values of the kpss test and critical test, in order to determine whether the time series is stable or not
    """
    kpss_test = kpss(timeseries)
    kpss_test_value = kpss_test[0]
    kpss_value = pd.DataFrame({key:value for key,value in kpss_test[3].items()},index = [0])
    kpss_value = pd.DataFrame(kpss_value)
    kpss_critical_value = kpss_value['10%'][0]
    return kpss_test_value, kpss_critical_value

def acorr_ljungbox_(timeseries):
    """

    :param timeseries: time series that aims to analyse
    :return: the values of the acorr ljungbox_test, in order to determine whether the time series is random or not
    """
    a = acorr_ljungbox(timeseries, lags=1)
    return a[1][0] ### return 检验结果的 p_value值


def model_makesense_determinate (total_dataset):
    """

    :param total_dataset: the dataset that contains analysis data
    :return: transform the umstabled dataset to a new one that is stabled.
    """
    adf_test_value, adf_critical_value = adf_(total_dataset.value)
    kpss_test_value, kpss_critical_value = kpss_(total_dataset.value)
    if adf_test_value > adf_critical_value and kpss_test_value > kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
        print 'Unstabled Value' #原始序列绝对不平稳
        total_dataset["value_diff"] = total_dataset["value"] - total_dataset["value"].shift(1)
        ######
        adf_test_value2, adf_critical_value2 = adf_(total_dataset.value)
        kpss_test_value2, kpss_critical_value2 = kpss_(total_dataset.value)

        if adf_test_value2 < adf_critical_value2 and kpss_test_value2 < kpss_critical_value2:
            print 'Stabled After Diff' #一阶差分后平稳
            a = acorr_ljungbox_(total_dataset.value_diff)
            if a < 0.05:
                total_dataset.value = total_dataset.value_diff ###由于差分之后时间序列平稳，用差分后的值代替value进行分析；






                # list_r = circulation_file_predict_origin_features_select_methods(total_dataset)
            # else: break
    #
    # # ######补充一个能够自动判断不平稳数据后经过在多次处理后 再循环判断，一直到数据平稳的过程
    # #
    # #
    # #
    # if adf_test_value > adf_critical_value and kpss_test_value < kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
    #     print '趋势平稳--去除趋势后序列严格平稳'
    #
    # if adf_test_value < adf_critical_value and kpss_test_value > kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
    #     print '差分平稳，利用差分可使序列平稳'
    #     total_dataset["value_diff"] = total_dataset["value"] - total_dataset["value"].shift(1)
    #     adf_test2 =  unitroot_adf(total_dataset.value_diff)
    #     # adfuller_value2= pd.DataFrame({key:value for key,value in adf_test[4].items()},index = [0])
    #     # adfuller_value = pd.DataFrame(adfuller_value)
    #     print adf_test2
    #
    # elif adf_test_value < adf_critical_value and kpss_test_value < kpss_critical_value: ##说明值小于任何一个%，也就是说序列是不平稳序列，需要进行差分处理
    #     print '绝对平稳'
    #     a = acorr_ljungbox_(total_dataset.value_diff)
    #     if a < 0.05:
    #         list_r = circulation_file_predict_origin_features_select_methods(total_dataset)
    #     else: break
    #     print t
    return total_dataset
    #t 用来判断是否是平稳和随机的
    #
