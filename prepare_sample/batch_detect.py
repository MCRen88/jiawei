#!/usr/bin/env python
# -*- coding=utf-8 -*-
"""
批量检测异常
"""
import os
from os.path import dirname, join
def gen_label_online(df, anom_type, N_sigma=3):
	"""
	anom_type: string, optional,{"global", "local"}
	"""
	df1 = df.copy()
	if anom_type is "local":
		val_std = df1["value"] / df1["value"].max()

		diff = val_std.shift(1) - val_std
		dfff_mean = diff.mean()
		dfff_std = diff.std()

		df1.loc[diff > dfff_mean + N_sigma * dfff_std, "label"] = 1
		df1.loc[diff < dfff_mean - N_sigma * dfff_std, "label"] = 1
		return df1

	if anom_type == "global":
		global_mean = df1["value"].mean()
		global_std = df1["value"].std()

		global_max = global_mean + N_sigma * global_std
		global_min = max(0.00000000001, global_mean - N_sigma * global_std)

		df1.loc[df1["value"] > global_max, "label"] = 1
		df1.loc[df1["value"] < global_min, "label"] = 1	
	return df1


if __name__=="__main__":
	import pandas as pd
	abs_path = os.path.realpath(__file__)
	df = pd.read_csv(join(dirname(abs_path),"../data/ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/A4Benchmark-TS1.csv"))

	df1 = gen_label_online(df, "global")
	print(df1[df1.label == 1].shape)

	df1 = gen_label_online(df, "local")
	print(df1[df1.label == 1].shape)