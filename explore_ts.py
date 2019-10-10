# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
from os import listdir,makedirs
from os.path import join,dirname
import json

import pandas as pd
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from visualize.plot_ts import plot_hist
from settings import Config_json,get_user_data_dir
from utils import savePNG,millisec_to_str

N_color = 10

DAY_SECONDS = 1440 * 60

FIGURE_SIZE = (16, 7)



def run():
    root_dir = get_user_data_dir()
    original_path = join(root_dir, "706_dnm_tmp_3ZFwT#sum_iUserNum_300#20190620_16Days_valid6D_pipeline_test_input_node_test_data.csv")
    df = pd.read_csv(original_path)
    print(df.head())
    df["timestamp"] = df["timestamp"].map(millisec_to_str)

    pic_path = join("/Users/stellazhao/research_space/jiawei", "tmp/pic/")
    
    line_id_list = np.unique(df.line_id)

    for l_id in line_id_list:
        df_slice = df[df.line_id == l_id].copy()
        print(df_slice.shape)
        plt = plot_hist(df_slice, detect_days = 2, plot_day_index=[1,7], anom_col = "label" , value_col = "point", freq = 300)
        savePNG(plt, targetDir=join(pic_path, "%s.png" % l_id))


if __name__ == "__main__":
    run()
