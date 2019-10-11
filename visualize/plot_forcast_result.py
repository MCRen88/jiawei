#!/usr/bin/env python
# -*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve




def Precision_Recall_Curve(training_data_anomaly, anomaly_score_train,test_data_anomaly, anomaly_score_test):
    """
    Obtain values of Precision and Recall of the prediction;
    Draw Precision_Recall_Curve

    :param y_true: True binary labels.
                If labels are not either {-1, 1} or {0, 1}, then pos_label should be explicitly given.
    :param y_pred_proba: Estimated probabilities or decision function.
    :param pos_label: The label of the positive class.
    """
    # def Precision_Recall_assessed_value(precision, recall,draw_type)
    # P-R图
    precision_train, recall_train, threshold_train = precision_recall_curve(training_data_anomaly, anomaly_score_train)
    precision_test, recall_test, threshold_test = precision_recall_curve(test_data_anomaly, anomaly_score_test)

    print(precision_train)
    print(recall_train)
    print(threshold_train)
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


def plot_auc(auc_score, precision, recall, label=None):
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

def anomaly_score_view_predict(dataset):

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

