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
from visualize.plot_ts import plot_hist, anomaly_view
from settings import Config_json,get_user_data_dir
from utils import savePNG,millisec_to_str
import os
from sklearn.metrics import precision_recall_curve,classification_report, confusion_matrix
from visualize.plot_forcast_result import anomaly_score_plot_hist
from evaluate.train_test_evaluate import prediction_evaluate
N_color = 10

DAY_SECONDS = 1440 * 60

FIGURE_SIZE = (16, 7)

from os.path import dirname, join, exists, isdir, split, isfile, abspath
import pickle as pkl
def savePklto(py_obj, targetDir):
    print("[savePklto]%s" % targetDir)
    with open(targetDir, "wb") as f:
        pkl.dump(py_obj, f)
    return


def loadPklfrom(sourceDir):
    print("[loadPklfrom]%s" % sourceDir)
    with open(sourceDir, "rb") as f:
        py_obj = pkl.load(f)
    return py_obj

def gen_pic():
    """

    :return: the plots of the multiple id dataset  based on the final date
    """
    root_dir = get_user_data_dir()

    original_path = join(root_dir, "706_dnm_tmp_3ZFwT#sum_iUserNum_300#20190620_16Days_valid6D_pipeline_test_input_node_train_data.csv")
    df = pd.read_csv(original_path)


    df["timestamps"] = df["timestamps"].map(millisec_to_str)
    #
    pic_path = join("/Users/xumiaochun/jiawei", "tmp/pic_valid/result_train/")

    line_id_list = np.unique(df.line_id)

    for l_id in line_id_list:
        l_id_list = l_id.split("valid")
        VALID_DAY = l_id_list[-1].replace("D", "")
        # print (VALID_DAY)
        if int(VALID_DAY) <10:
            continue
        df_slice = df[df.line_id == l_id].copy()
        print(df_slice.shape)
        plt = plot_hist(df_slice, detect_days = 2, plot_day_index=[1,7], anom_col = "label" , value_col = "point", freq = 300)
        savePNG(plt, targetDir=join(pic_path, "%s.png" % l_id))


def extract_selected_id_name():
    """

    :return: the id name of the selected pics in a file
    """
    root_dir = get_user_data_dir()
    original_path = join(root_dir, "706_dnm_tmp_3ZFwT#sum_iUserNum_300#20190620_16Days_valid6D_pipeline_test_input_node_train_data.csv")
    df = pd.read_csv(original_path)
    # print(df.head(2))
    # print df.columns.tolist()
    df["timestamp"] = df["timestamp"].map(millisec_to_str)
    #
    pic_path = join("/Users/xumiaochun/jiawei", "tmp/pic_valid_select/")

    file_dir = pic_path
    i = 1
    a = os.walk(file_dir)
    b = None
    for root, dirs, files in os.walk(file_dir):
        print(i)
        i += 1
        print(root) #当前目录路径
        print(dirs) #当前路径下所有子目录
        print(files) #当前路径下所有非目录子文件

    return b

def select_data():
    """

    :return: concat the data on selected id name
    """
    selected_id = pd.read_csv('/Users/xumiaochun/jiawei/tmp/selected_name_list2.csv')

    # root_dir = get_user_data_dir()
    # original_path = join(root_dir, "706_dnm_tmp_3ZFwT#sum_iUserNum_300#20190620_16Days_valid6D_pipeline_test_input_node_test_data")
    origin_dataset = pd.read_csv("/Users/xumiaochun/jiawei/tmp/data_multiple/706_dnm_tmp_3ZFwT#sum_iUserNum_300#20190620_16Days_valid6D_pipeline_test_input_node_test_data.csv")    # selected_id = pd.read_csv('/Users/xumiaochun/jiawei/tmp/selected_name_list2.csv')
    a0 = origin_dataset[origin_dataset['line_id'] == selected_id.selected_id[0]].reset_index(drop=True)
    if (a0.point[0] ==0 and a0.point[1] ==0 and a0.point[2] ==0):
        for i in range(3,len(a0)):
            if (a0.point[i] == 0):
                i = i+1
            elif (a0.point[i] !=0):
                k = i
                break
        a0 = a0.ix[k:].reset_index(drop=True)
    t = a0
    for j in range(1,len(selected_id)):
        id_name = selected_id.selected_id[j]
        a1 = origin_dataset[origin_dataset['line_id'] == id_name].reset_index(drop=True)
        if (a1.point[0] == 0 and a1.point[1] == 0 and a1.point[2] == 0):
            for i in range(3, len(a1)):
                if (a1.point[i] == 0):
                    i = i + 1
                elif (a1.point[i] != 0):
                    k = i
                    break
            a1 = a1.ix[k:].reset_index(drop=True)
        t = pd.concat([t,a1],axis=0)

    t.to_csv("/Users/xumiaochun/jiawei/tmp/aggregate_2.csv",index = False)


def check_result():
    wrong_pre = pd.read_csv ('/Users/xumiaochun/jiawei/new_check_dataset_wrongpre2__test.csv')
    origin_dataset = pd.read_csv("/Users/xumiaochun/jiawei/concat_test.csv")
    wrong_pre_id = np.unique(wrong_pre.line_id)
    wrong_pre_id = pd.DataFrame(wrong_pre_id,columns={"line_id"})
    # origin_dataset.rename(columns={"timestamps":"timestamp", "anomaly":"label","value":"point"}, inplace=True)

    id_name = wrong_pre_id.line_id[0]
    a0 = origin_dataset[origin_dataset['line_id'] == id_name].reset_index(drop=True)
    t = a0
    for j in range(1,len(wrong_pre_id)):
        id_name = wrong_pre_id.line_id[j]
        a1 = origin_dataset[origin_dataset['line_id'] == id_name].reset_index(drop=True)
        t = pd.concat([t, a1], axis=0)
    df = t
    pic_path = join("/Users/xumiaochun/jiawei", "tmp/pic_check/result_test2/")

    line_id_list = np.unique(df.line_id)

    for l_id in line_id_list:
        l_id_list = l_id.split("valid")
        VALID_DAY = l_id_list[-1].replace("D", "")
        if int(VALID_DAY) <10:
            continue
        df_slice = df[df.line_id == l_id].copy()
        # plt = anomaly_view(df_slice)
        print(df_slice.shape)
        plt = plot_hist(df_slice, detect_days = 2, plot_day_index=[1,7], anom_col = "label" , pre_anom_col = "y_pred", value_col = "point", freq = 300)
        savePNG(plt, targetDir=join(pic_path, "%s.png" % l_id))

def concat():
    ori_dataset = pd.read_csv("/Users/xumiaochun/jiawei/dataset_test.csv")
    y_pred_df = loadPklfrom("/Users/xumiaochun/jiawei/y_pred_test.csv")
    y_pred_score_df = loadPklfrom("/Users/xumiaochun/jiawei/anomaly_score_test.csv")
    y_pred_df = pd.DataFrame(y_pred_df,columns={'y_pred_test'})
    y_pred_score_df = pd.DataFrame(y_pred_score_df,columns={'anomaly_score_test'})

    # selected_id = pd.read_csv('/Users/xumiaochun/jiawei/tmp/selected_name_list2.csv')
    id_ = np.unique(ori_dataset.line_id)
    # id_ = selected_id.selected_id

    ori_dataset["y_pred"] = 99
    ori_dataset["y_pred_score"] = 99


    id_name = id_[0]
    id_dataset = ori_dataset[ori_dataset['line_id'] == id_name]
    id_dataset = id_dataset.reset_index(drop=True)
    DAY_PNT = len(id_dataset.loc[id_dataset['Date'] == id_dataset['Date'].ix[int(len(id_dataset) / 2)]])
    lenth = len(id_dataset)
    k = 0
    y_pred = y_pred_df.ix[0:  2 * DAY_PNT].reset_index(drop=True)
    y_score = y_pred_score_df.ix[0: 2 * DAY_PNT].reset_index(drop=True)


    for t in range(lenth - 2 * DAY_PNT-1, lenth):
        id_dataset.y_pred.ix[t] = y_pred.y_pred_test.ix[(t - lenth + 2 * DAY_PNT+1)]
        id_dataset.y_pred_score.ix[t] = y_score.anomaly_score_test.ix[(t - lenth + 2 * DAY_PNT+1)] ##!!
    dataset_toprint = id_dataset

    for j in range(1,len(id_)):
    # for j in range(1, 3):
        id_name = id_[j]
        id_dataset = ori_dataset[ori_dataset['line_id'] == id_name]
        id_dataset = id_dataset.reset_index(drop=True)
        DAY_PNT = len(id_dataset.loc[id_dataset['Date'] == id_dataset['Date'].ix[int(len(id_dataset) / 2)]])
        lenth = len(id_dataset)
        k = 1
        y_pred = y_pred_df.ix[2 * k * DAY_PNT: 2 * k * DAY_PNT + 2 * DAY_PNT].reset_index(drop=True)
        y_score = y_pred_score_df.ix[2 * k * DAY_PNT:2 * k * DAY_PNT + 2 * DAY_PNT].reset_index(drop=True)

        for t in range(lenth - 2 * DAY_PNT-1, lenth):
            id_dataset.y_pred.ix[t] = y_pred.y_pred_test.ix[(t - lenth + 2 * DAY_PNT+1)]
            id_dataset.y_pred_score.ix[t] = y_score.anomaly_score_test.ix[(t - lenth + 2 * DAY_PNT+1)] ##!!
        k = k + 1
        dataset_toprint = pd.concat([dataset_toprint,id_dataset],axis = 0)
    dataset_toprint = dataset_toprint.reset_index(drop=True)
    dataset_toprint = pd.DataFrame(dataset_toprint)
    a_df = dataset_toprint[dataset_toprint.y_pred_score != 99]
    a_c_df =  a_df[a_df.anomaly != a_df.y_pred]
    print (a_c_df)
    #
    dataset_toprint.to_csv("/Users/xumiaochun/jiawei/tmp/data/multiple/concat_test222",index=False)

def concat_train():

    ori_dataset = pd.read_csv("/Users/xumiaochun/jiawei/dataset_train.csv")
    y_pred_df = loadPklfrom("/Users/xumiaochun/jiawei/y_pred_train.csv")
    y_pred_score_df = loadPklfrom("/Users/xumiaochun/jiawei/anomaly_score_train.csv")
    y_pred_df = pd.DataFrame(y_pred_df,columns={'y_pred_train'})
    y_pred_score_df = pd.DataFrame(y_pred_score_df,columns={'anomaly_score_train'})
    id_ = np.unique(ori_dataset.line_id)

    ori_dataset["y_pred"] = 99
    ori_dataset["y_pred_score"] = 99


    id_name = id_[0]
    id_dataset = ori_dataset[ori_dataset['line_id'] == id_name]
    id_dataset = id_dataset.reset_index(drop=True)
    DAY_PNT = len(id_dataset.loc[id_dataset['Date'] == id_dataset['Date'].ix[int(len(id_dataset) / 2)]])
    lenth = len(id_dataset)
    k = 0
    y_pred = y_pred_df.ix[0:  2 * DAY_PNT].reset_index(drop=True)
    y_score = y_pred_score_df.ix[0: 2 * DAY_PNT].reset_index(drop=True)


    for t in range(lenth - 2 * DAY_PNT-1, lenth):
        id_dataset.y_pred.ix[t] = y_pred.y_pred_train.ix[(t - lenth + 2 * DAY_PNT+1)]
        id_dataset.y_pred_score.ix[t] = y_score.anomaly_score_train.ix[(t - lenth + 2 * DAY_PNT+1)]
    dataset_toprint = id_dataset

    for j in range(1,len(id_)):
    # for j in range(1, 3):
        id_name = id_[j]
        id_dataset = ori_dataset[ori_dataset['line_id'] == id_name]
        id_dataset = id_dataset.reset_index(drop=True)
        DAY_PNT = len(id_dataset.loc[id_dataset['Date'] == id_dataset['Date'].ix[int(len(id_dataset) / 2)]])
        lenth = len(id_dataset)
        k = 1
        y_pred = y_pred_df.ix[2 * k * DAY_PNT: 2 * k * DAY_PNT + 2 * DAY_PNT].reset_index(drop=True)
        y_score = y_pred_score_df.ix[2 * k * DAY_PNT:2 * k * DAY_PNT + 2 * DAY_PNT].reset_index(drop=True)

        for t in range(lenth - 2 * DAY_PNT-1, lenth):
            id_dataset.y_pred.ix[t] = y_pred.y_pred_train.ix[(t - lenth + 2 * DAY_PNT+1)]
            id_dataset.y_pred_score.ix[t] = y_score.anomaly_score_train.ix[(t - lenth + 2 * DAY_PNT+1)] ##!!
        k = k + 1
        dataset_toprint = pd.concat([dataset_toprint,id_dataset],axis = 0)
    dataset_toprint = dataset_toprint.reset_index(drop=True)
    dataset_toprint = pd.DataFrame(dataset_toprint)
    a_df = dataset_toprint[dataset_toprint.y_pred_score != 99]
    a_c_df =  a_df[a_df.anomaly != a_df.y_pred]
    print (a_c_df)
    dataset_toprint.to_csv("/Users/xumiaochun/jiawei/tmp/data/multiple/concat_train222",index= False)

def check():
    pic_path = join("/Users/xumiaochun/jiawei", "tmp/pic_check/result_test2/score/")
    dff = pd.read_csv("/Users/xumiaochun/jiawei/concat_test.csv")
    dff.rename(columns={"timestamps":"timestamp", "anomaly":"label","value":"point"}, inplace=True)
    line_id_list = np.unique(dff.line_id)
    for l_id in line_id_list:
        l_id_list = l_id.split("valid")
        VALID_DAY = l_id_list[-1].replace("D", "")
        if int(VALID_DAY) <10:
            continue
        df_slice = dff[dff.line_id == l_id].copy()
        # plt = anomaly_view(df_slice)
        plt = anomaly_score_plot_hist(df_slice, detect_days = 2, plot_day_index=[1,7], anom_col = "label" ,pre_anom_col = "y_pred", value_col = "point", freq = 60)
        # print (df_slice.head())
        savePNG(plt, targetDir=join(pic_path, "%s.png" % l_id))


if __name__ == "__main__":
    # labels = loadPklfrom("/Users/xumiaochun/jiawei/labels_train2.csv")
    # y_pred = loadPklfrom("/Users/xumiaochun/jiawei/y_pred_train2.csv")
    # concat_test = pd.read_csv("/Users/xumiaochun/jiawei/concat_test.csv")
    # print (y_pred.head())
    # # view_dataset = concat_test[concat_test["y_pred"] != 99]
    # prediction_evaluate(labels.anomaly,y_pred.y_pred_train)

    concat()
    concat_train()