# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
from os import listdir,makedirs
from os.path import join,dirname
import json
import time
import datetime



import pandas as pd
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
# from visualize.plot_ts import plot_hist
from settings import Config_json,get_user_data_dir
N_color = 10

DAY_SECONDS = 1440 * 60

FIGURE_SIZE = (16, 7)


def format_to_millisec(dt):
    return int((time.mktime(time.strptime(dt, "%Y-%m-%d %H:%M:%S"))))


def millisec_to_str(sec):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(sec)))



def get_normal_path(a):
    # a = str(a)
    a = repr(a)
    a = a.replace('\\', '').replace('x00', '').replace('\'', '').replace('\"', '')
    if not a.startswith("/"):
        a = "/" + "/".join(a.split("/")[1:])
    return a

def mkdir_p(path):
    try:
        makedirs(path)
        print("[mkdir_p]", path)
    except OSError as exc:
        # Python >2.5 (except OSError, exc: for Python <2.5)
        # if exc.errno == errno.EEXIST and isdir(path):
        pass
        # else:
        #     raise


def savePNG(plt, targetDir=None, **kwargs):
    if targetDir is None:
        return
    file_path = get_normal_path(targetDir)
    if not file_path.startswith("/"):
        file_path = "/" + "/".join(file_path.split("/")[1:])
    mkdir_p(dirname(file_path))
    print("[savePNG]%s" % file_path)
    plt.savefig(file_path, **kwargs)
    return