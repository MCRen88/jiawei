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
	time_series_abs_energy
	time_series_absolute_sum_of_changes
	time_series_variance_larger_than_std
	time_series_count_above_mean
	time_series_count_below_mean
	time_series_mean_abs_change
	time_series_percentage_of_reoccurring_datapoints_to_all_datapoints
	time_series_ratio_value_number_to_time_series_length
	time_series_sum_of_reoccurring_data_points
	time_series_sum_of_reoccurring_values
    time_series_range
    time_series_mean_change
    time_series_has_duplicate
    time_series_has_duplicate_min
    time_series_longest_strike_above_mean
    time_series_longest_strike_below_mean
    time_series_first_location_of_maximum
    time_series_first_location_of_minimum
    time_series_last_location_of_maximum
    time_series_last_location_of_minimum
    		
	abs_energy
	friedrich_coefficients （param (list) {“m”: x, “r”: y, “coeff”: z} x为正整数，是多项式拟合的最高阶数，y是正实数，用于计算均值的分位数，z为正整数，多项式的第几项。）
	ratio_beyond_r_sigma：远离 x 的平均值大于 r * std(x)(s r sigma)的值的比率。
	large_standard_deviation：标准差是否为数据范围的r倍（bool）
    number_peaks
    fft_aggregated：返回绝对傅里叶变换后的光谱质心、峰度、偏度等值（pandas.Series）
    ratio_beyond_r_sigma： 译：偏离x的平均值大于r * std(x)(so r sigma)的值的比率。
    
	。。。
- 异常检测相关特征(feature_anom.py)：
	- 局部异常特征：
		time_series_moving_average， 
		time_series_weighted_moving_average
		time_series_exponential_weighted_moving_average
		time_series_double_exponential_weighted_moving_average
		
		binned_entropy（局部分组求熵）
		quantile：计算 x 的 q 分位数。这是大于 x 的有序值的前 q\%的 x 值
		change_quantiles（译：给定区间的时序数据描述统计，然后在这个区间里计算时序数据的均值、绝对值、连续变化值。（浮点数））
		number_crossing_m
		energy_ratio_by_chunks 分块局部熵比率（将时序数据分块后，计算目标块数据的熵与全体的熵比率。当数据不够均分时，会将多余的数据在前面的块中散布。（浮点数））
        agg_linear_trend    基于分块时序聚合值的线性回归
        number_cwt_peaks
                               		
	- 周期性异常特征：
		time_series_periodic_feature 。。。
		
- 曲线形态特征(feature_pattern.py)：
	time_series_autocorrelation
	time_series_coefficient_of_variation
	time_series_value_distribution
	time_series_daily_parts_value_distribution
	time_series_daily_parts_value_distribution_with_threshold
	time_series_binned_entropy
	
	approximate_entropy
	sample_entropy
	cwt_coefficients
	fft_coefficient
	ar_coefficient
	energy_ratio_by_chunks
	cid_ce
	partial_autocorrelatio
	agg_autocorrelation
	symmetry_looking
	time_reversal_asymmetry_statistic
	C3
	spkt_welch_density：译：该特征计算器估计不同频率下时间序列x的交叉功率谱密度。为此，首先将时间序列从时域转移到频域。
                       特征计算器返回不同频率的功率谱。
    linear_trend 译：线性回归分析
	linear_trend_timewise
	fft_coefficient
	
	
	
	---unsure
    
    index_mass_quantile：计算某分位数对应的索引值（pandas.Series）
    max_langevin_fixed_point

    
    --delete
    value_count：译：计算时间序列x中value出现的次数
    range_count 计算区间[min，max]内的观测值的个数。
    augmented_dickey_fuller 衡量时序数据的平稳性  
                             




########



