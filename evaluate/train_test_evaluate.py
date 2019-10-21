# -*- coding: UTF-8 -*-


from sklearn.metrics import classification_report,confusion_matrix,f1_score,average_precision_score


from time_series_detector.algorithm.gbdt import *

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

from os.path import dirname, join, exists, isdir, split, isfile, abspath
import pickle as pkl


# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import sys
sys.setrecursionlimit(1000000)


import pandas as pd
import numpy as np

from visualize.plot_ts import plot_hist, anomaly_view

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




def plot_tree(clf, title="example"):
    from sklearn.tree import export_graphviz
    import graphviz
    dot_data = export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(title)
    pass



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


##########################################--- main ----#####################################################################
##########################################--- main ----#####################################################################
if __name__ == "__main__":


    warnings.filterwarnings("ignore")
    window = 14

    #train dataset
    dataset_train = pd.read_csv('/Users/xumiaochun/jiawei/tmp/selected_data.csv')
    dataset_train.rename(columns={"timestamp":"timestamps", "label":"anomaly","point":"value"}, inplace=True)
    dataset_train = characteristics_format_match(dataset_train) #total_dataset中含有两新列----Date和Hour_Minute
    selected_id = np.unique(dataset_train.line_id)

    #test dataset
    dataset_test = pd.read_csv('/Users/xumiaochun/jiawei/tmp/selected_data_test.csv')
    dataset_test.rename(columns={"timestamp":"timestamps", "label":"anomaly","point":"value"}, inplace=True)
    dataset_test = characteristics_format_match(dataset_test) #total_dataset中含有两新列----Date和Hour_Minute



    selected_features_train, labels_train = selecte_festures (selected_id,dataset_train)
    selected_features_test, labels_test = selecte_festures(selected_id,dataset_test)


    y_pred_train, anomaly_score_train, y_pred_test, anomaly_score_test = data_modeling_gbdt(selected_features_train, labels_train.anomaly,selected_features_test)



    print ("\n\nlabels_train.anomaly, y_pred_train\n", (classification_report(labels_train.anomaly, y_pred_train)))
    print ("\n\nlabels_test.anomaly, y_pred_test\n", (classification_report(labels_test.anomaly, y_pred_test)))


    tn, fp, fn, tp = confusion_matrix(labels_train.anomaly, y_pred_train).ravel()
    confusion_train = confusion_matrix(labels_train.anomaly, y_pred_train).ravel()
    print ("confusion_train2",confusion_train)
    print ("\nconfusion value__train:\n", tn, fp, fn, tp)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(labels_test.anomaly, y_pred_test).ravel()
    print ("\nconfusion value__test:\n", tn_test, fp_test, fn_test, tp_test)
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
    y_pred_test = y_pred_test.reset_index(drop = True)

    anomaly_score_test = pd.DataFrame(anomaly_score_test, columns={'anomaly_score_test'})
    anomaly_score_test = anomaly_score_test.reset_index(drop = True)

    labels_test = labels_test.reset_index(drop = True)
    new_check_dataset_test = pd.concat([labels_test,y_pred_test],axis=1)
    new_check_dataset_test = pd.concat([new_check_dataset_test,anomaly_score_test],axis=1)
    new_check_dataset_wrongpre_test = new_check_dataset_test.loc[new_check_dataset_test.anomaly != new_check_dataset_test.y_pred_test]
    new_check_dataset_wrongpre_test = new_check_dataset_wrongpre_test.reset_index(drop = True)



    for j in range(0, len(selected_id)):
        id_name = selected_id[j]
        id_dataset = new_check_dataset_wrongpre_train[new_check_dataset_wrongpre_train['line_id'] == id_name]
        f1_score_value = f1_score(labels_train.anomaly, y_pred_train, average='binary')
        if f1_score_value < 0.7 and f1_score_value > 0.5:
            print("0.5<f1_score<0.7__train", "id_name:", id_name, anomaly_view(id_dataset, id_name))
        if f1_score_value < 0.5:
            print("f1_score<0.5__train", "id_name:", id_name, anomaly_view(id_dataset, id_name))

    #
    for j in range(0, len(selected_id)):
        id_name = selected_id[j]
        id_dataset = new_check_dataset_wrongpre_test[new_check_dataset_wrongpre_test['line_id'] == id_name]
        f1_score_value = f1_score(labels_test.anomaly, y_pred_test, average='binary')
        if f1_score_value<0.7 and f1_score_value >0.5:
            print ("0.5<f1_score<0.7__test","id_name:",id_name,anomaly_view(id_dataset,id_name))
        if f1_score_value<0.5:
            print ("f1_score<0.5__test","id_name:",id_name,anomaly_view(id_dataset,id_name))


