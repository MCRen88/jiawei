# -*- coding: UTF-8 -*-

import csv
import re
# import matplotlib.pylab as plt

# from inspect import signature
import datetime
import warnings
import time
import json, ast
from datetime import timedelta
import json, ast

# import pywt #导入PyWavelets

# import feature
# import feature.extraction
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report,confusion_matrix,f1_score,average_precision_score

from sklearn.model_selection import train_test_split


# from time_series_detector.feature.extraction import *
from time_series_detector.algorithm.gbdt import *
import time_series_detector.algorithm.gbdt
# from sklearn.cross_validation import train_test_split
# from data_pre_processing.stable_random_test import model_makesense_determinate
from data_preprocessing.data_format_match import characteristics_format_match
from visualize.plot_ts import anomaly_view, plot_hist
from feature.features_calculate_select \
    import combine_features_calculate,sliding_window, combine_features_calculate\
    , features_selected_ExtraTreesClassifier,selected_columns_names, feature_extraction,cal_features_based_on_id
from visualize.plot_forcast_result import Precision_Recall_Curve2, plot_auc, anomaly_predict_view\
    , anomaly_score_view_predict, anomaly_score_view_date,preci_rec
from visualize.plot_stable_view import value_stable_determinate_view
from sklearn.metrics import confusion_matrix
from utils import savePNG,millisec_to_str

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
from settings import Config_json,get_user_data_dir
from utils import savePNG,millisec_to_str
import os
from sklearn import metrics
from os.path import dirname, join, exists, isdir, split, isfile, abspath
import pickle as pkl


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


N_color = 10

DAY_SECONDS = 1440 * 60

FIGURE_SIZE = (16, 7)




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


def data_modeling_gbdt(x_train,y_train,x_test):
    '''
    :param x_train: selected features dataset
    :param y_train: Label data (or Flag)
    :return: y_train_pred(not DF), represents whether the predicte value would be 0 or 1(anomaly)
     anomaly_score(not DF), represents the percentage to be anomaly_train.
    '''

    # clf = GradientBoostingClassifier()
    # clf.fit(x_train, y_train)
    # joblib.dump(clf, 'train_model_result_pipeline.m')  ##save clf model
    # for cnt, tree in enumerate(clf.estimators_):
    #     plot_tree(clf=tree[0], title="example_tree_%d" % cnt)

    clf = "./train_model_result_pipeline.m"
    if exists("./train_model_result_pipeline.m"):
        clf = joblib.load("./train_model_result_pipeline.m")
    else:
        clf = GradientBoostingClassifier()
        clf.fit(x_train, y_train)
        joblib.dump(clf, 'train_model_result_pipeline.m')

    features_importance = clf.feature_importances_

    y_pred_train = clf.predict(x_train)
    anomaly_score_train = clf.predict_proba(x_train)[:, 1]

    y_pred_test = clf.predict(x_test)
    anomaly_score_test = clf.predict_proba(x_test)[:, 1]

    print ("\nfeatures_importance_modelling\n",features_importance)
    print ("\nfeature\n",x_train.columns.tolist())

    return y_pred_train,anomaly_score_train,y_pred_test,anomaly_score_test





def circulation_file_predict_origin_features_select_methods(total_dataset):

    #思路
    ##先进行绘图：1、曲线和异常点的图；2、观察某个随机日期的数据（自行输入日期），得到前一天；上一周和本周的异常情况
    ##先进行特征计算
    #再进行特征选择
    #在进行数据集 split；从win+7day开始--从win+7day开始的数据集
    #在进行预测算法；
    #对异常的预测情况的precision_recall的画图；
    #单独的precision和recall图
    #结果根据时间窗添加到原始total——dataset中,对比flag为异常和预测为异常的情况


    ##先进行绘图：1、曲线和异常点的图；2、观察某个随机日期的数据（自行输入日期），得到前一天；上一周和本周的异常情况
    #1、曲线和异常点的图

    anomaly_view(total_dataset)#观察异常点和整体时序走向
    ##特征计算
    x_features_selected,y_calculate, selected_features_name = feature_extraction(total_dataset,window)


    # #特征选择
    #在进行数据集 split；从win+7day开始--从win+7day开始的数据集
    new_dataset = total_dataset.iloc[win_sli-1:,:]
    new_dataset = new_dataset.reset_index(drop= True)
    training_data, test_data = train_test_split(new_dataset, test_size = 0.3, shuffle=False)
    x_train_selected, x_test_selected = train_test_split(x_features_selected, test_size = 0.3, shuffle=False)
    y_train_selected,y_test_selected = train_test_split(y_calculate, test_size = 0.3, shuffle=False)




    # #在进行预测算法；
    y_pred_train, anomaly_score_train = data_modeling_gbdt(x_train_selected,training_data.anomaly)
    y_pred_test, anomaly_score_test = data_modeling_gbdt(x_test_selected,test_data.anomaly)

    predict_report_train = classification_report (training_data.anomaly,y_pred_train, labels=[1, 2, 3])
    predict_report_test = classification_report (test_data.anomaly,y_pred_test, labels=[1, 2, 3])




    anomaly_pred = []
    anomaly_pred.extend(y_pred_train)
    anomaly_pred.extend(y_pred_test)
    anomaly_pred = pd.DataFrame(anomaly_pred,columns={'anomaly_pred'})


    anomaly_pred_score = []
    anomaly_pred_score.extend(anomaly_score_train)
    anomaly_pred_score.extend(anomaly_score_test)
    anomaly_pred_score = pd.DataFrame(anomaly_pred_score,columns={'anomaly_pred_score'})
    #
    new_dataset = pd.concat([new_dataset,anomaly_pred,anomaly_pred_score],axis=1)  #new dataset 包含了的数据包括原始数据集的属性+预测值和预测分数；并ignore window+7*day_pnt的值


    anomaly_predict_view(new_dataset) ##compare the difference between anomaly_flag and anomaly_predict
    anomaly_score_view_predict(new_dataset) ##观察整条曲线的anomaly_score和anomaly——flag的情况

    anomaly_score_view_date(new_dataset,'2015-01-07') ##观察某一天anomaly_score和anomaly——flag的情况

# #对异常的预测情况的precision_recall的画图；
    Precision_Recall_Curve(training_data.anomaly, anomaly_score_train,test_data.anomaly, anomaly_score_test)




def plot_tree(clf, title="example"):
    from sklearn.tree import export_graphviz
    import graphviz
    dot_data = export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(title)
    pass

#
# def run():
#     """
#
#     :return: the plots of the multiple id dataset  based on the final date
#     """
#     root_dir = get_user_data_dir()
#
#     original_path = join(root_dir, "706_dnm_tmp_3ZFwT#sum_iUserNum_300#20190620_16Days_valid6D_pipeline_test_input_node_train_data.csv")
#     df = pd.read_csv(original_path)
#     # print(df.head())
#     df["timestamp"] = df["timestamp"].map(millisec_to_str)
#     #
#     pic_path = join("/Users/xumiaochun/jiawei", "tmp/pic_valid/")
#
#     line_id_list = np.unique(df.line_id)
#
#     for l_id in line_id_list:
#         l_id_list = l_id.split("valid")
#         VALID_DAY = l_id_list[-1].replace("D", "")
#         print VALID_DAY
#         if int(VALID_DAY) <10:
#             continue
#         df_slice = df[df.line_id == l_id].copy()
#         print(df_slice.shape)
#         plt = plot_hist(df_slice, detect_days = 2, plot_day_index=[1,7], anom_col = "label" , value_col = "point", freq = 300)
#         savePNG(plt, targetDir=join(pic_path, "%s.png" % l_id))


def selecte_festures(selected_id,dataset):
    new_dataset = []
    calculate_features = []
    labels = []
    for j in range(0,len(selected_id)):
    # for j in range(0, 2):
        id_name = selected_id[j]
        id_dataset = dataset[dataset['line_id'] == id_name]
        cal_features,y_calculate = cal_features_based_on_id(id_dataset,window,id_name)
        calculate_features.append(cal_features)
        labels.append(y_calculate)

    name = "{}".format(dataset)

    calculate_features = pd.concat(calculate_features)
    labels = pd.concat(labels)
    calculate_features = calculate_features.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    calculate_features = calculate_features.ix[:, ~((calculate_features == 1).all() | (calculate_features == 0).all()) | (calculate_features == 'inf').all()]
    print ("calculate_features of {} 的大小【计算完后的特征】".format(name),calculate_features.shape)
    selected_features = features_selected_ExtraTreesClassifier(calculate_features,labels.anomaly)
    print ("selected_features of {}的大小".format(name),selected_features.shape)
    return calculate_features,labels

def concat(ori_dataset,y_pred_df,y_pred_score_df):
    # ori_dataset = pd.read_csv("/Users/xumiaochun/jiawei/dataset_test.csv")
    # y_pred_df = pd.read_csv("/Users/xumiaochun/jiawei/y_pred_test.csv")
    # y_pred_score_df = pd.read_csv("/Users/xumiaochun/jiawei/anomaly_score_test.csv")
    id_ = np.unique(ori_dataset.line_id)
    ori_dataset["y_pred"] = 99
    ori_dataset["y_pred_score"] = 99
    k = 0

    id_name = id_[0]
    id_dataset = ori_dataset[ori_dataset['line_id'] == id_name]
    id_dataset = id_dataset.reset_index(drop=True)
    DAY_PNT = len(id_dataset.loc[id_dataset['Date'] == id_dataset['Date'].ix[int(len(id_dataset) / 2)]])
    lenth = len(id_dataset)
    k = 0
    y_pred = y_pred_df.ix[0:  2 * DAY_PNT - 1].reset_index(drop=True)
    y_score = y_pred_score_df.ix[0: 2 * DAY_PNT - 1].reset_index(drop=True)


    for t in range(lenth - 2 * DAY_PNT, lenth):
        id_dataset.y_pred.ix[t] = y_pred.y_pred_test.ix[(t - lenth + 2 * DAY_PNT)]
        id_dataset.y_pred_score.ix[t] = y_score.anomaly_score_test.ix[(t - lenth + 2 * DAY_PNT)]
    dataset_toprint = id_dataset
    a_df = id_dataset[id_dataset.y_pred_score != 99].copy()

    for j in range(1,len(id_)):
        id_name = id_[j]
        id_dataset = ori_dataset[ori_dataset['line_id'] == id_name]
        id_dataset = id_dataset.reset_index(drop=True)
        DAY_PNT = len(id_dataset.loc[id_dataset['Date'] == id_dataset['Date'].ix[int(len(id_dataset) / 2)]])
        lenth = len(id_dataset)
        k = 1
        y_pred = y_pred_df.ix[2 * k * DAY_PNT: 2 * k * DAY_PNT + 2 * DAY_PNT-1].reset_index(drop=True)
        y_score = y_pred_score_df.ix[2 * k * DAY_PNT:2 * k * DAY_PNT + 2 * DAY_PNT-1].reset_index(drop=True)

        for t in range(lenth - 2 * DAY_PNT, lenth):
            id_dataset.y_pred.ix[t] = y_pred.y_pred_test.ix[(t - lenth + 2 * DAY_PNT )]
            id_dataset.y_pred_score.ix[t] = y_score.anomaly_score_test.ix[(t - lenth + 2 * DAY_PNT )]
        k = k + 1
        dataset_toprint = pd.concat([dataset_toprint,id_dataset],axis = 0)

    dataset_toprint = pd.DataFrame(dataset_toprint)

    return dataset_toprint


def check_result(wrong_pre,origin_dataset,place = None):
    # wrong_pre = pd.read_csv ('/Users/xumiaochun/jiawei/new_check_dataset_wrongpre2__test.csv')
    # origin_dataset = pd.read_csv("/Users/xumiaochun/jiawei/concat_test.csv")
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
    pic_path = join("/Users/xumiaochun/jiawei", "tmp/pic_check/{}/".format(place))

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

##########################################--- main ----#####################################################################
##########################################--- main ----#####################################################################
if __name__ == "__main__":


    warnings.filterwarnings("ignore")
    window = 14

    #train dataset
    dataset_train = pd.read_csv('/Users/xumiaochun/jiawei/tmp/selected_data.csv')
    print ("原始数据集的数据集描述（行数，列数）_train:", dataset_train.shape, "\ncolumns_names:", dataset_train.columns.tolist())

    list_to_print = []
    dataset_train.rename(columns={"timestamp":"timestamps", "label":"anomaly","point":"value"}, inplace=True)
    dataset_train = characteristics_format_match(dataset_train) #total_dataset中含有两新列----Date和Hour_Minute
    selected_id = np.unique(dataset_train.line_id)
    print ("id数量:", len(selected_id))
    print ("进行时间处理后的数据集描述（行数，列数）_train：",dataset_train.shape,"\ncolumns_names:", dataset_train.columns.tolist())

    #test dataset
    dataset_test = pd.read_csv('/Users/xumiaochun/jiawei/tmp/selected_data_test.csv')
    print ("原始数据集的数据集描述（行数，列数）_test:", dataset_test.shape, "\ncolumns_names:", dataset_test.columns.tolist())
    dataset_test.rename(columns={"timestamp":"timestamps", "label":"anomaly","point":"value"}, inplace=True)
    dataset_test = characteristics_format_match(dataset_test) #total_dataset中含有两新列----Date和Hour_Minute
    print ("进行时间处理后的数据集描述（行数，列数）_test：",dataset_test.shape,"\ncolumns_names:", dataset_test.columns.tolist())



    selected_features_train, labels_train = selecte_festures (selected_id,dataset_train)
    selected_features_test, labels_test = selecte_festures(selected_id,dataset_test)


    y_pred_train, anomaly_score_train, y_pred_test, anomaly_score_test = data_modeling_gbdt(selected_features_train, labels_train.anomaly,selected_features_test)

    # predict_report_train = classification_report(labels_train.anomaly, y_pred_train, labels=[1,2])
    # predict_report_test = classification_report(labels_test.anomaly, y_pred_test, labels=[1,2])

    # print ("\npredict_report_train\n",predict_report_train)
    # print ("\npredict_report_test\n",predict_report_test)

    print ("\n\nlabels_train.anomaly, y_pred_train\n", (classification_report(labels_train.anomaly, y_pred_train)))
    print ("\n\nlabels_test.anomaly, y_pred_test\n", (classification_report(labels_test.anomaly, y_pred_test)))


    confusion_train = confusion_matrix(labels_train.anomaly, y_pred_train).ravel()
    print ("confusion_train2",confusion_train)
    confusion_test = confusion_matrix(labels_test.anomaly, y_pred_test).ravel()
    print  ("confusion_test2",confusion_test)
    Precision_Recall_Curve2(labels_train.anomaly, anomaly_score_train)
    Precision_Recall_Curve2(labels_test.anomaly, anomaly_score_test)


    y_pred_train =pd.DataFrame(y_pred_train, columns={'y_pred_train'})
    anomaly_score_train = pd.DataFrame(anomaly_score_train, columns={'anomaly_score_train'})
    labels_train = labels_train.reset_index(drop = True)
    y_pred_train = y_pred_train.reset_index(drop = True)
    anomaly_score_train = anomaly_score_train.reset_index(drop = True)
    new_check_dataset_train = pd.concat([labels_train,y_pred_train],axis=1)
    new_check_dataset_train = pd.concat([new_check_dataset_train,anomaly_score_train],axis=1)
    new_check_dataset_wrongpre_train = new_check_dataset_train.loc[new_check_dataset_train.anomaly != new_check_dataset_train.y_pred_train]
    new_check_dataset_wrongpre_train = new_check_dataset_wrongpre_train.reset_index(drop = True)

    print (new_check_dataset_wrongpre_train.head(2),new_check_dataset_wrongpre_train.columns.tolist())
    print ('\ncontain wrong pred id include__train:\n',np.unique(new_check_dataset_wrongpre_train.line_id))
    print ('\nwrong_pred__train\n',new_check_dataset_wrongpre_train)
    new_check_dataset_wrongpre_train.to_csv("new_check/new_check_dataset_wrongpre2__train.csv",index=False)


    y_pred_test =pd.DataFrame(y_pred_test, columns={'y_pred_test'})
    anomaly_score_test = pd.DataFrame(anomaly_score_test, columns={'anomaly_score_test'})
    labels_test = labels_test.reset_index(drop = True)
    y_pred_test = y_pred_test.reset_index(drop = True)
    anomaly_score_test = anomaly_score_test.reset_index(drop = True)
    new_check_dataset_test = pd.concat([labels_test,y_pred_test],axis=1)
    new_check_dataset_test = pd.concat([new_check_dataset_test,anomaly_score_test],axis=1)
    new_check_dataset_wrongpre_test = new_check_dataset_test.loc[new_check_dataset_test.anomaly != new_check_dataset_test.y_pred_test]
    new_check_dataset_wrongpre_test = new_check_dataset_wrongpre_test.reset_index(drop = True)

    print (new_check_dataset_wrongpre_test.head(2),new_check_dataset_wrongpre_test.columns.tolist())
    print ('\ncontain wrong pred id include__test:\n',np.unique(new_check_dataset_wrongpre_test.line_id))
    print ('\nwrong_pred__test\n',new_check_dataset_wrongpre_test)
    new_check_dataset_wrongpre_test.to_csv("new_check/new_check_dataset_wrongpre2__test.csv",index=False)

    ##对应时间戳

    dataset_train.to_csv("new_check/dataset_train.csv",index=False)
    y_pred_train.to_csv("new_check/y_pred_train.csv",index=False)
    anomaly_score_train.to_csv("new_check/anomaly_score_train.csv",index=False)
    dataset_test.to_csv("new_check/dataset_test.csv",index=False)
    y_pred_test.to_csv("new_check/y_pred_test.csv",index=False)
    anomaly_score_test.to_csv("new_check/anomaly_score_test.csv",index=False)

    # dataset_train_t = include_pre_dataset(dataset_train, labels_train,y_pred_train,anomaly_score_train)
    # dataset_test_t = include_pre_dataset(dataset_test, labels_train, y_pred_test,anomaly_score_test)
    # dataset_train = pd.concat([dataset_train,y_pred_train,anomaly_score_train],axis=1)
    # dataset_test = pd.concat([dataset_test,y_pred_test,anomaly_score_test],axis=1)
    # dataset_train_t.to_csv("dataset_train_t.csv",index=False)
    # dataset_test_t.to_csv("dataset_test_t.csv",index=False)
    dataset_toprint_train = concat(dataset_train, y_pred_train, anomaly_score_train)
    dataset_toprint_test = concat(dataset_test, y_pred_test, anomaly_score_test)
    dataset_toprint_train.to_csv("new_check/concat_train_2re.csv",index = False)
    dataset_toprint_test.to_csv("new_check/concat_test_2re.csv", index=False)


    df_train = dataset_toprint_train.copy()
    y_pred = []
    y_true = []
    line_id = np.unique(dataset_toprint_train.line_id.values)
    for l in line_id:
        df_sclie = df_train[df_train.line_id == l].copy()
        y_true.extend(list(df_sclie.anomaly.values[-400:]))
        y_pred.extend(list(df_sclie.y_pred.values[-400:]))
    report_train = classification_report(y_true, y_pred)
    print("report_train",report_train)
    confusion_train = confusion_matrix(y_true, y_pred)
    print("report_train",confusion_train)

    df_test = dataset_toprint_test.copy()
    y_pred = []
    y_true = []
    line_id = np.unique(dataset_toprint_test.line_id.values)

    for l in line_id:
        df_sclie = df_test[df_test.line_id == l].copy()
        y_true.extend(list(df_sclie.anomaly.values[-400:]))
        y_pred.extend(list(df_sclie.y_pred.values[-400:]))
    report_test = classification_report(y_true, y_pred)
    print("report_test",report_test)
    confusion_test = confusion_matrix(y_true, y_pred)
    print("report_test",confusion_test)
    # Precision_Recall_Curve2(labels_train.anomaly, anomaly_score_train)
    # ffff(labels.anomaly, anomaly_score_train)

    for j in range(0, len(selected_id)):
        # for j in range(0, 1):

        id_name = selected_id[j]
        id_dataset = new_check_dataset_wrongpre_train[new_check_dataset_wrongpre_train['line_id'] == id_name]
        f1_score_value = f1_score(labels_train.anomaly, y_pred_train, average='binary')
        if f1_score_value < 0.7 and f1_score_value > 0.5:
            print("0.5<f1_score<0.7__train", "id_name:", id_name, anomaly_view(id_dataset, id_name))
        if f1_score_value < 0.5:
            print("f1_score<0.5__train", "id_name:", id_name, anomaly_view(id_dataset, id_name))

    #
    for j in range(0, len(selected_id)):
    # for j in range(0, 1):

        id_name = selected_id[j]
        id_dataset = new_check_dataset_wrongpre_test[new_check_dataset_wrongpre_test['line_id'] == id_name]
        f1_score_value = f1_score(labels_test.anomaly, y_pred_test, average='binary')
        if f1_score_value<0.7 and f1_score_value >0.5:
            print ("0.5<f1_score<0.7__test","id_name:",id_name,anomaly_view(id_dataset,id_name))
        if f1_score_value<0.5:
            print ("f1_score<0.5__test","id_name:",id_name,anomaly_view(id_dataset,id_name))


        ##add 整体pr曲线
    ##案例分析：fscore少于0.5的曲线样本画图



    # df = new_check_dataset_wrongpre
    # print(df.head())
    # df["timestamp"] = df["timestamp"].map(millisec_to_str)
    #
    # pic_path = join("/Users/xumiaochun/jiawei", "tmp/pic_valid/")




# #数据特征提取要based整个数据集combine_features_calculate（所有id）；；
#
#
#
# ########——————————后续检测
#     # anomaly_view(total_dataset)#观察异常点和整体时序走向 （未进行数据平稳处理前）
#     # #判断序列的平稳性
#     # value_stable_determinate_view(total_dataset)
#
#     # total_dataset= model_makesense_determinate (total_dataset)
#
    # list_r = circulation_file_predict_origin_features_select_methods(total_dataset)
#
#     #
#     # #
#     # # global x_features_selected
#     # warnings.filterwarnings("ignore")
#     # window = 14
#     # DEFAULT_WINDOW = 14
#     # k = pd.read_csv('data/ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/totoal_file_and_name.csv')
#     # k = pd.DataFrame(k)
#     # list_to_print = []
#     # for i in range(0,len(k)):
#     #     location = k["location"].ix[i]
#     #     filename = k["filename"].ix[i]
#     #     total_dataset = pd.read_csv('{}'.format(location))
#     #     if i >= 124:
#     #         total_dataset.rename(columns={"timestamp":"timestamps", "is_anomaly":"anomaly"}, inplace=True)
#     #
#     #     total_dataset = characteristics_format_match(total_dataset) #total_dataset中含有两新列----Date和Hour_Minute
#     #     DAY_PNT = len(total_dataset.loc[total_dataset['Date'] == total_dataset['Date'].ix[len(total_dataset)/2]])
#     #     lenth_total_dataset = len(total_dataset)
#     #     win_sli = window + 7 * DAY_PNT
#     #     lenth_new_dataset = len(total_dataset.ix[win_sli-1:]) #真正有特征值部分的数据集
#     #
#     #     training_data, test_data = train_test_split(total_dataset.ix[win_sli-1:], test_size = 0.3, shuffle=False)
#     #     train_ = total_dataset.ix[win_sli-1:int(lenth_new_dataset*0.7)+win_sli-1]
#     #     test_ = total_dataset.ix[int(lenth_new_dataset*0.7)+win_sli-1:]
#     #
#     #     if win_sli < int(len(total_dataset)*0.3) and len(test_.loc[test_['anomaly'] == 1]) > 0 and len(train_.loc[train_['anomaly'] == 1]) > 0: ####（要加入判断时间序列的分析有没有价值的判断方法）
#     #         anomaly_view(total_dataset)#观察异常点和整体时序走向 （未进行数据平稳处理前）
#     #         # #判断序列的平稳性
#     #         value_stable_determinate(total_dataset)
#     #
#     #         total_dataset= model_makesense_determinate (total_dataset)
#     #
#     #         list_r = circulation_file_predict_origin_features_select_methods(total_dataset)
#     #
#     #         break ##只跑一个数据集
#
#
