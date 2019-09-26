def gen_label_online(df, anom_type):

	start, start2, end = df["timestamp"].values[[0, 1, -1]]
	freq = format_to_millisec(start2) - format_to_millisec(start)
	# ano = AnomalyInjection(df, start, end, freq , 0.1)
	# ano.predict()
	# df1 = ano.df.copy()
	df1 = df.copy()
	# df1["label"].astype = int
	if anom_type is None:
		df1["prob"] = abs(df1["anomaly_value"] - df1["prediction"])/df1['std']
		df1.loc[df1["prob"] > 3., "label"] = 1

		global_mean = df1["value"].mean()
		global_std = df1["value"].std()

		global_max = global_mean + 3 * global_std
		global_min = max(1, global_mean - 3 * global_std)
		if global_mean > 100:
			global_min = 10

		df1.loc[df1["anomaly_value"] > global_max, "label"] = 1
		df1.loc[df1["anomaly_value"] < global_min, "label"] = 1

		val_std = df1["anomaly_value"] / df1["anomaly_value"].max()

		diff = val_std.shift(1) - val_std
		dfff_mean = diff.mean()
		dfff_std = diff.std()

		df1.loc[diff > dfff_mean + 3 * dfff_std, "label"] = 1
		df1.loc[diff < dfff_mean - 3 * dfff_std, "label"] = 1


		return df1

	if anom_type == "global":
		global_mean = df1["value"].mean()
		global_std = df1["value"].std()

		global_max = global_mean + 3 * global_std
		global_min = max(0.00000000001, global_mean - 3 * global_std)

		df1.loc[df1["anomaly_value"] > global_max, "label"] = 1
		df1.loc[df1["anomaly_value"] < global_min, "label"] = 1
		

	if anom_type == "global&thresh":
		global_min  = 0.5
		df1.loc[df1["anomaly_value"] < global_min, "label"] = 1
	
	return df1