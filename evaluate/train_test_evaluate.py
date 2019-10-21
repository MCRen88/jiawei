# -*- coding: UTF-8 -*-

from sklearn.metrics import classification_report,confusion_matrix,f1_score,average_precision_score

print("\n\nlabels_train.anomaly, y_pred_train\n", (classification_report(labels_train.anomaly, y_pred_train)))
print("\n\nlabels_test.anomaly, y_pred_test\n", (classification_report(labels_test.anomaly, y_pred_test)))

confusion_train = confusion_matrix(labels_train.anomaly, y_pred_train).ravel()
print("confusion_train2", confusion_train)
confusion_test = confusion_matrix(labels_test.anomaly, y_pred_test).ravel()
print("confusion_test2", confusion_test)
