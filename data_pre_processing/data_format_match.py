#!/usr/bin/env python
# -*- coding=utf-8 -*-

import time
import pandas as pd


def millisec_to_str(sec):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(sec)))

def characteristics_format_match(total_dataset):
    # total_dataset.rename(columns={"timestamp":"timestamps", "is_anomaly":"anomaly","point":"value"}, inplace=True)
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
    return total_dataset