## 说明

feature: 存放时间序列特征相关的计算逻辑
主要分为三类：
- 统计量( feature_stat.py)：
	time_series_maximum
	time_series_minimum
	time_series_mean
	time_series_variance
	time_series_standard_deviation
	time_series_skewness
	time_series_kurtosis
	time_series_median
	time_series_sum_values
	time_series_range
	time_series_abs_energy
	time_series_absolute_sum_of_changes
	time_series_variance_larger_than_std
	time_series_count_above_mean
	time_series_count_below_mean
	time_series_longest_strike_above_mean(x)
	time_series_longest_strike_below_mean(x)
	time_series_mean_abs_change(x)
	time_series_mean_change(x)
	time_series_percentage_of_reoccurring_datapoints_to_all_datapoints(x)
	time_series_ratio_value_number_to_time_series_length(x)
	time_series_sum_of_reoccurring_data_points(x)
	time_series_sum_of_reoccurring_values(x)
	
	。。。
- 异常检测相关特征(feature_anom.py)：
	- 局部异常特征：
		time_series_moving_average， 
		time_series_weighted_moving_average
		time_series_exponential_weighted_moving_average
		time_series_double_exponential_weighted_moving_average
	- 周期性异常特征：
		time_series_periodic_feature 。。。
- 曲线形态特征(feature_pattern.py)：
	time_series_autocorrelation
	time_series_coefficient_of_variation
	time_series_value_distribution
	time_series_daily_parts_value_distribution
	time_series_daily_parts_value_distribution_with_threshold
	time_series_binned_entropy
	。。。
