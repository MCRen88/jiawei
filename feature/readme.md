## 说明

feature: 存放时间序列特征相关的计算逻辑
主要分为三类：
- 统计量( feature_stat.py)：
	time_series_maximum
	time_series_minimum
	。。。
- 异常检测相关特征：
	- 局部异常特征：
		time_series_moving_average， 
		time_series_exponential_weighted_moving_average
		。。。
	- 周期性异常特征：
		time_series_periodic_feature 。。。
- 曲线形态特征：
	time_series_autocorrelation
	time_series_coefficient_of_variation
	。。。
