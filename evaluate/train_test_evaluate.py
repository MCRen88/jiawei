# -*- coding: UTF-8 -*-




import warnings
warnings.filterwarnings("ignore")
import sys
sys.setrecursionlimit(1000000)



import pandas as pd
import numpy as np
from visualize.plot_ts import anomaly_view
from sklearn.metrics import classification_report, confusion_matrix,f1_score


def prediction_evaluate(y_true,y_pred):
    """

    :param y_true: label of the data
    :param y_pred: predict value of the data
    :return:
    """
    classification_report_result = classification_report(y_true, y_pred)
    confusion_matrix_result = confusion_matrix(labels_train.anomaly, y_pred_train).ravel()
    print ("\nclassification_report_result\n",classification_report_result,"\n\nconfusion_matrix_result\n",confusion_matrix_result)

def poor_pred_id_dataset_view(labels_with_id,y_pred,y_pred_score):

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


