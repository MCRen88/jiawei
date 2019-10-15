# -*- coding: utf-8 -*-
"""
批量检测所有曲线
"""
from os.path import join

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import average_precision_score

from src.visualization.utils import plot_error_rate
from utils import loadPklfrom,  saveDF, savePklto, mkdir_p, savePNG
from settings import Config_json,get_user_data_dir


config_json = Config_json()
input_dir =  config_json.get_config("train_data")

## output
output_dir = config_json.get_config("model_dir")
output_raw_res_format = join(output_dir, "feature_mode%s", "%s_summary_prob.csv")


RNG = np.random.RandomState(42)
N_jobs = 10
CV_DEBUG = ShuffleSplit(n_splits=2, test_size=0.5, random_state=RNG)
CV = ShuffleSplit(n_splits=5, test_size=0.2, random_state=RNG)

ensemble_clfs_params = {'n_estimators': [100],
                              'max_depth': [3, 5],
                              'min_samples_split': [15],
                              'min_samples_leaf': [10, ],
                              'max_features': ['auto'],
                              'loss': ['deviance', ],
                              }

ensemble_clfs_params_debug = {'n_estimators': [10],
                              'max_depth': [3, ],
                              'min_samples_split': [15, ],
                              'min_samples_leaf': [10, ],
                              'max_features': ['auto'],
                              'loss': ['deviance', ],
                              }

clf_dict = {
    "model": ensemble.GradientBoostingClassifier(random_state=RNG, verbose=1),
    "params": ensemble_clfs_params,
    "params_debug": ensemble_clfs_params_debug
}


def model_train(X, Y):
    CLF = GridSearchCV(clf_dict["model"], clf_dict["params"], scoring=scoring_clf,
                       verbose=6, cv=CV, n_jobs=N_jobs)
    Y = np.array(Y.values, dtype=int).squeeze()
    CLF.fit(X, Y.reshape(-1, 1))
    print 'best params:\n', CLF.best_params_
    mean_scores = np.array(CLF.cv_results_['mean_test_score'])
    print 'mean score', mean_scores
    print 'best score', CLF.best_score_
    print 'worst score', np.min(mean_scores)
    clf = CLF.best_estimator_
    return clf, X.columns


def scoring_clf(clf, x, y):
    pred_prob = model_predict(clf, x)
    assert pred_prob.size == y.size
    average_precision = average_precision_score(y, pred_prob,)
    return average_precision


def model_predict(clf, X):
    return clf.predict_proba(X)[:, 1]


def heldout_score(clf, X, Y):
    """compute deviance scores on ``X_test`` and ``y_test``. """
    n_estimators, n_classes = clf.estimators_.shape
    score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X)):
        score[i] = clf.loss_(Y, y_pred)
    return score


def train(feature_mode):
    [X_train0, Y_train0, _, _] = loadPklfrom(join(input_dir, "feature_mode%s" % feature_mode, "train_set_sample.pkl"))
    clf, f_list = model_train(X_train0, Y_train0)
    savePklto(f_list, join(output_dir, "feature_mode%s" % feature_mode, "gbdt_feature_list.pkl"))

    score = heldout_score(clf, X_train0, Y_train0)
    plt = plot_error_rate(score)

    savePNG(plt, join(output_dir, "feature_mode%s" % feature_mode, "error_rate.png"))
    savePklto(clf, join(output_dir, "feature_mode%s" % feature_mode, "gbdt.pkl"))
    return


def serving(feature_mode, data_set="train"):
    print "*********load %s set*********" % data_set
    [X_train, Y_train, value_train, ids_train] = loadPklfrom(join(input_dir, "feature_mode%s" % feature_mode, "%s_set.pkl" % data_set))
    print "Got # %s  lines " % (np.unique(np.array(ids_train)).size)
    clf = loadPklfrom(join(output_dir, "feature_mode%s" % feature_mode, "gbdt.pkl"))
    # feature_list = loadPklfrom(join(output_dir, "feature_mode%s" % feature_mode, "gbdt_feature_list.pkl"))
    xns = np.array(X_train.values, dtype=float)
    feature_nan_ind = np.any(~np.isfinite(xns), axis=1)
    train_pred = np.zeros(xns.shape[0])
    train_pred[~feature_nan_ind] = model_predict(clf, xns[~feature_nan_ind])
    train_sum = pd.DataFrame({"value": value_train,
                                "id": ids_train,
                                "label": Y_train,
                                "prob": train_pred,
                                "value_period_history_detector(xs, history_xs)": X_train["origin_period_history_detector"],
                                "value_period_history_trend_detector(xs, history_xs)": X_train["origin_period_history_trend_detector"],
                                # "value_period_curve_branch_score": X_train["value_period_curve_branch_score"],
                                # "value_smooth_curve_branch_score": X_train["value_smooth_curve_branch_score"],
                              })
    train_sum["value_period_correlation_distance(xs, history_xs_piece)"] = 0
    saveDF(train_sum, output_raw_res_format % (feature_mode, data_set))
    return


if __name__ == "__main__":
    train(feature_mode=11)
    serving(feature_mode=11, data_set="train")
    serving(feature_mode=11, data_set="test")


