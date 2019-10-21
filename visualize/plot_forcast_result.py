#!/usr/bin/env python
# -*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import classification_report, confusion_matrix,f1_score
from visualize.plot_ts import anomaly_view

# def ffff(y, scores):
#     n_classes = len(y)
#
#     # Add noisy features to make the problem harder
#     random_state = np.random.RandomState(0)
#     # n_samples, n_features = X.shape
#     # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#     #
#     # # shuffle and split training and test sets
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
#     #                                                     random_state=0)
#     #
#     # # Learn to predict each class against the other
#     # classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
#     #                                          random_state=random_state))
#     # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
#
#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y, scores)
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y, scores)
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#     plt.figure()
#     lw = 2
#     plt.plot(fpr[2], tpr[2], color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.show()




def Precision_Recall_Curve(training_data_anomaly, anomaly_score_train,test_data_anomaly, anomaly_score_test):
    # """
    # Obtain values of Precision and Recall of the prediction;
    # Draw Precision_Recall_Curve
    #
    # :param y_true: True binary labels.
    #             If labels are not either {-1, 1} or {0, 1}, then pos_label should be explicitly given.
    # :param y_pred_proba: Estimated probabilities or decision function.
    # :param pos_label: The label of the positive class.
    # """
    """

    :param training_data_anomaly: label of training dataset
    :param anomaly_score_train: predicted anomaly score of training dataset
    :param test_data_anomaly: label of test dataset
    :param anomaly_score_test: predicted anomaly score of test dataset
    :return: precision,recall,threshold of training dataset and test dataset, as well as to draw the Precision-Recall plot
    """
    # def Precision_Recall_assessed_value(precision, recall,draw_type)
    # P-R图
    precision_train, recall_train, threshold_train = precision_recall_curve(training_data_anomaly, anomaly_score_train)
    precision_test, recall_test, threshold_test = precision_recall_curve(test_data_anomaly, anomaly_score_test)

    # plt.plot(recall,precision)
    plt.plot(recall_train,precision_train,label = 'train_precision_recall')
    # plt.plot(recall_test,precision_test,label = 'test_precision_recall')

    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc = 'best')
    plt.show()

#'Precision and Recall of Training Data
    plt.plot(precision_train,label = 'precision_train')
    plt.plot(recall_train,label = 'recall_train')
    plt.title('Precision and Recall of Training Data')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc = 'best')
    plt.show()

#Precision and Recall of Test Data
    plt.plot(precision_test,label = 'precision_test')
    plt.plot(recall_test,label = 'recall_test')
    plt.title('Precision and Recall of Test Data')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc = 'best')
    plt.show()

    return precision_train, recall_train, threshold_train ,precision_test, recall_test, threshold_test

# def true_predict_curve(y,x):
#     y_pred = clf.predict(x)
#     plt.plot(y, color='blue', label='true data')
#     plt.plot(y_pred, color='red',label='predict data')
#     plt.legend(loc='best')
#     plt.title('Comparison of True and Predict Anomaly Curve')
#     plt.show(block=False)
def Precision_Recall_Curve2(training_data_anomaly, anomaly_score_train):
    # """
    # Obtain values of Precision and Recall of the prediction;
    # Draw Precision_Recall_Curve
    #
    # :param y_true: True binary labels.
    #             If labels are not either {-1, 1} or {0, 1}, then pos_label should be explicitly given.
    # :param y_pred_proba: Estimated probabilities or decision function.
    # :param pos_label: The label of the positive class.
    # """
    """

    :param training_data_anomaly: label of training dataset
    :param anomaly_score_train: predicted anomaly score of training dataset
    :param test_data_anomaly: label of test dataset
    :param anomaly_score_test: predicted anomaly score of test dataset
    :return: precision,recall,threshold of training dataset and test dataset, as well as to draw the Precision-Recall plot
    """
    # def Precision_Recall_assessed_value(precision, recall,draw_type)
    # P-R图
    precision_train, recall_train, threshold_train = precision_recall_curve(training_data_anomaly, anomaly_score_train)
    print ("\n\nprecision_train\n",precision_train,"\n\nrecall_train\n",recall_train,"\n\nthreshold_train\n",threshold_train)
    plt.plot(recall_train,precision_train,label = 'train_precision_recall')

    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc = 'best')
    plt.show()

#'Precision and Recall of Training Data
    plt.plot(precision_train,label = 'precision_train')
    plt.plot(recall_train,label = 'recall_train')
    # precision_train.plot()
    # recall_train.plot()
    plt.title('Precision and Recall of Training Data')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc = 'best')
    plt.show()
    return precision_train, recall_train, threshold_train


def plot_auc(auc_score, precision, recall, label=None):
    """

    :param auc_score: auc_score of the predict label
    :param precision: precision of the prediction
    :param recall: recall of the prediction
    :param label: label
    :return: auc curve
    """
    plot_pr.figure(num=None, figsize=(6, 5))
    plot_pr.xlim([0.0, 1.0])
    plot_pr.ylim([0.0, 1.0])
    # pylplot_prob.xlabel('Recall')
    plot_pr.ylabel('Precision')
    plot_pr.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
    plot_pr.fill_between(recall, precision, alpha=0.5)
    plot_pr.grid(True, linestyle='-', color='0.75')
    plot_pr.plot(recall, precision, lw=1)
    plot_pr.show()


def anomaly_score_view_date(dataset,target_date):
    """

    :param total_dataset: the dataset that contains analysis data
    :param target_date: a date among the dataset, to determine the difference of the predict value and the label
    :return: a plot that determine whether there are differences between the predict value and the label based on date
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    total_dataset_Date_index = dataset
    total_dataset_Date_index = total_dataset_Date_index.set_index('Date')
    total_dataset_Date_index.index = pd.DatetimeIndex(total_dataset_Date_index.index)


    line_name1 = 'Line_{}'.format(target_date)
    line1 = total_dataset_Date_index.loc[target_date] ##12。24为基准
    ax.plot(line1.Hour_Minute, line1.value, color = 'red',  label=line_name1,alpha = 0.4)
    if len(line1.loc[line1['anomaly']== 1])>0:
        line_anomaly = line1.loc[line1['anomaly']== 1]
        ax.scatter(np.array(line_anomaly.Hour_Minute),line_anomaly.value, marker='+',color = 'red', label='{}'.format(line_name1))

    ax2 = ax.twinx()
    ax2.plot(line1.Hour_Minute, line1.anomaly_pred_score, label = 'Anomaly Score')
    ax.legend(loc = 'best')
    ax2.legend(loc = 'best')
    plt.xticks(rotation=30)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Value changed")
    ax2.set_ylabel("Probability Anomaly Score")
    ax2.set_ylim(0, 1)
    plt.xticks(rotation = 30)
    plt.title('Anomaly Score Date View')
    plt.show()


def anomaly_score_plot_hist(df, detect_days = 2, plot_day_index=[1,7], anom_col = None ,pre_anom_col = None, value_col = "value", freq = 60):
    """
    detect_days: 检测时间长度，以天为单位
    freq: 时序间隔，以秒为单位
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

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

## --加入预测部分
    # ax2 = ax.twinx()
    #
    # pre_anom_df_slice = df[df[pre_anom_col] == 1][value_col]
    # info = "%s#%spts" % (pre_anom_col, pre_anom_df_slice.shape[0])
    # if pre_anom_df_slice.shape[0] >= 1:
    #     pre_anom_df_slice.plot(c="red", linewidth=5, style='+', label=info)
    #

    ax2 = ax.twinx()
    ax2.plot(df.Hour_Minute, df.anomaly_pred_score, label = 'Anomaly Score')
    ax.legend(loc = 'best')
    ax2.legend(loc = 'best')
    plt.xticks(rotation=30)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Value changed")
    ax2.set_ylabel("Probability Anomaly Score")
    ax2.set_ylim(0, 1)
    plt.xticks(rotation = 30)
    plt.title('Anomaly Score Date View')
    plt.show()


    return plt


def anomaly_score_view_predict(dataset):
    """

    :param total_dataset: the dataset that contains analysis data
    :return: a plot that shows predict value and the label
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    anomaly_flag = dataset.set_index("anomaly")
    anomaly_flag = anomaly_flag.loc[1]

    plt.plot(pd.to_datetime(dataset.timestamps),dataset.value, color = 'black',alpha = 0.3)
    plt.scatter(pd.to_datetime(np.array(anomaly_flag.timestamps)),anomaly_flag.value, color='black',marker='+', label='Anomaly_Flag',alpha = 0.8)
    plt.xticks(rotation=30)

    ax2 = ax.twinx()
    ax2.plot(pd.to_datetime(dataset.timestamps), dataset.anomaly_pred_score, label = 'Anomaly Score')
    ax.legend(loc = 'best')
    ax2.legend(loc = 'best')

    ax.set_xlabel("Date (h)")
    ax.set_ylabel("Value changed")
    ax2.set_ylabel("Probability Anomaly Score")
    ax2.set_ylim(0, 1)
    plt.title('Anomaly Score Predict View')

    plt.show()



def anomaly_predict_view(dataset):
    """

    :param total_dataset: the dataset that contains analysis data
    :return: a plot that determine whether there are differences between the predict value and the label
    """
    anomaly_flag = dataset.set_index("anomaly")
    anomaly_flag = anomaly_flag.loc[1]
    anomaly_pridict = dataset.set_index("anomaly_pred")
    anomaly_pridict = anomaly_pridict.loc[1]

    plt.plot(pd.to_datetime(dataset.timestamps),dataset.value, color = 'black',alpha = 0.3)
    plt.scatter(pd.to_datetime(np.array(anomaly_flag.timestamps)),anomaly_flag.value, color='black',marker='+', label='Anomaly_Flag',alpha = 0.8)
    plt.scatter(pd.to_datetime(np.array(anomaly_pridict.timestamps)),anomaly_pridict.value, color='orange',marker='^', label='Anomaly_Predict',alpha = 0.8)

    plt.legend(loc= 'best',fontsize= 5)
    plt.xticks(rotation=30)
    plt.title('Anomaly Predict View')

    plt.show()


def preci_rec(dataset):
    precision, recall, _ = precision_recall_curve(dataset.anomaly, dataset.anomaly_score_train)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    average_precision = average_precision_score(dataset.anomaly, dataset.anomaly_score_train)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    plt.show()

def poor_pred_id_dataset_view(labels_with_id,y_pred,y_pred_score):
    '''

    :param labels_with_id: the dataframe that contains labels and id information
    :param y_pred: predict data
    :param y_pred_score: predict score data
    :return:
    '''
    y_pred_test =pd.DataFrame(y_pred, columns={'y_pred_test'})
    anomaly_score_test = pd.DataFrame(y_pred_score, columns={'anomaly_score_test'})
    labels_test = labels_with_id.reset_index(drop = True)
    y_pred_test = y_pred_test.reset_index(drop = True)
    anomaly_score_test = anomaly_score_test.reset_index(drop = True)
    new_check_dataset_test = pd.concat([labels_test,y_pred_test],axis=1)
    new_check_dataset_test = pd.concat([new_check_dataset_test,anomaly_score_test],axis=1)
    new_check_dataset_wrongpre_test = new_check_dataset_test.loc[new_check_dataset_test.anomaly != new_check_dataset_test.y_pred_test]
    new_check_dataset_wrongpre_test = new_check_dataset_wrongpre_test.reset_index(drop = True)
    selected_id = np.unique(new_check_dataset_wrongpre_test.line_id)
    for j in range(0, len(selected_id)):
        id_name = selected_id[j]
        id_dataset = new_check_dataset_wrongpre_test[new_check_dataset_wrongpre_test['line_id'] == id_name]
        f1_score_value = f1_score(labels_test.anomaly, y_pred_test, average='binary')
        if f1_score_value<0.7 and f1_score_value >0.5:
            print ("0.5<f1_score<0.7__test","id_name:",id_name,anomaly_view(id_dataset,id_name))
        if f1_score_value<0.5:
            print ("f1_score<0.5__test","id_name:",id_name,anomaly_view(id_dataset,id_name))
