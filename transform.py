# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import sys
sys.setrecursionlimit(1000000)
import os
from os import listdir,makedirs
from os.path import join,dirname
import json

import pandas as pd
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt


from visualize.plot_ts import plot_hist
from feature import feature_anom 

from utils import savePNG,millisec_to_str
from settings import Config_json, get_user_data_dir, PROJECT_DIR
from curve_list import c

N_color = 10

DAY_SECONDS = 1440 * 60

FIGURE_SIZE = (16, 7)

train_pnts = 402

window_size = 288*2


feature_list = ["time_series_moving_average",
           "time_series_weighted_moving_average",
           "time_series_exponential_weighted_moving_average",
           "time_series_double_exponential_weighted_moving_average",
           # "time_series_periodic_features", #####append 的问题？？
           # "binned_entropy",
           # "quantile",
           # "change_quantiles",
           # "number_crossing_m",
           # "energy_ratio_by_chunks",
           # "agg_linear_trend",
           # "number_cwt_peaks"
           ]


def get_window_samples(df):
    samples = []
    n_samples, _ = df.shape
    values = df.point.values
    labels = df.label.values[-train_pnts:].tolist()
    for indx in range(train_pnts):
        slice_window = values[n_samples - train_pnts + indx + 1 - window_size :n_samples - train_pnts + indx + 1]
        samples.append(slice_window)
    return (samples , labels)


def cal_features(samples):
    features = []
    for sample in samples:
        feature_set = []
        for f in feature_list:
            # print("begin to cal ", f)
            func = getattr(feature_anom, f)
            format_sample = [[],[],[],[],sample]
            feature_set.append(func(format_sample))
        features.append(feature_set)
    features = pd.DataFrame(features) 
    return features




def run():
    """

    :return: the plots of the multiple id dataset  based on the final date
    """
    config_json = Config_json()
    train_dir =  config_json.get_config("train_data")
    original_path =  config_json.get_config("original_path")
    df = pd.read_csv(original_path)
    df["timestamp"] = df["timestamp"].map(millisec_to_str)


    feature_all_curves = []
    labels_all_curves = []
    for l_id in c:
      df_slice = df[df.line_id == l_id].copy()

      samples,label = get_window_samples(df_slice)
      features = cal_features(samples)

      labels_all_curves.append(label)
      feature_all_curves.append(features)
      # plt = plot_hist(df_slice, detect_days = 2, plot_day_index=[1,7], anom_col = "label" , value_col = "point", freq = 300)
      # savePNG(plt, targetDir=join(pic_path, "%s.png" % l_id))
    feature_all_curves = pd.concat(feature_all_curves)
    labels_all_curves= pd.concat(labels_all_curves)
    saveDF(feature_all_curves, join(train_dir, "features.csv"))
    saveDF(labels_all_curves, join(train_dir, "labels.csv"))
    return 


if __name__ == "__main__":
    run()
