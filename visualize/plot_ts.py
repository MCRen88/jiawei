# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import sys   
sys.setrecursionlimit(1000000)

from os import listdir
from os.path import join
import json

import pandas as pd
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

N_color = 10

DAY_SECONDS = 1440 * 60

FIGURE_SIZE = (16, 7)




def plot_hist(df, detect_days = 2, plot_day_index=[1,7], anom_col = None , value_col = "value", freq = 60):
    """
    detect_days: 检测时间长度，以天为单位
    freq: 时序间隔，以秒为单位
    """
    day_pnts = int( DAY_SECONDS / freq)
    print("day_pnts",day_pnts)
    detect_points = detect_days * day_pnts
    print("detect_points",detect_points)

    plt.figure(figsize=FIGURE_SIZE)
    df = df.sort_values("timestamp")

    df["timestamp"] = df["timestamp"].map(pd.to_datetime)
    print(df.head())
    for shift_ in plot_day_index:
        df["shift_%sd" % shift_] = df[value_col].shift(day_pnts * shift_)
    df = df.iloc[-detect_points:, :]
    df = df.set_index("timestamp")
    for shift_ in plot_day_index:
        df["shift_%sd" % shift_].plot(label="%sdays_beefore"% shift_, alpha=0.8)
    df[value_col].plot(label="today", alpha=0.8)
    plt.legend()
    if anom_col is None:
        return plt

    anom_df_slice = df[df[anom_col] == 1][value_col]
    info = "%s#%spts" % (anom_col, anom_df_slice.shape[0])
    if anom_df_slice.shape[0] >= 1:
        anom_df_slice.plot(c="g", linewidth=5, style='>', label=info)
    return plt



def anomaly_view(dataset):
    """

    :param total_dataset: the dataset that contains analysis data
    :return: a plot shows the value changed and the anomaly value
    """
    y_train_anomaly_dataset_selcted = dataset.set_index("anomaly")
    y_train_anomaly_dataset_selcted = y_train_anomaly_dataset_selcted.loc[1]

    plt.plot(pd.to_datetime(dataset.timestamps),dataset.value, color = 'black',alpha = 0.3)
    plt.scatter(pd.to_datetime(np.array(y_train_anomaly_dataset_selcted.timestamps)),y_train_anomaly_dataset_selcted.value, color='red',marker='+', label='Maxmin_value_anomaly')
    plt.legend(loc= 'best',fontsize= 5)
    plt.xticks(rotation=30)
    plt.title('Value and Anomaly view')# give plot a title
    plt.show()