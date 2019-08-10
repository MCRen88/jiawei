import pandas as pd
from time_series_detector.algorithm.gbdt import *
from time_series_detector.common.tsd_common import DEFAULT_WINDOW, split_time_series
from time_series_detector.common.tsd_common import DEFAULT_WINDOW, split_time_series

DEFAULT_WINDOW = 6
test_data = pd.read_csv('data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv')

gbdt = Gbdt()
split_ts = split_time_series(list(test_data.value))
normalized_split_value = tsd_common.normalize_time_series(split_ts)


# ex_feature = extract_features(test_data.value, 6)
features = []
for index in test_data:
    if is_standard_time_series(normalized_split_value, 6):
        print ("y")
        temp = []
        temp.append(feature_service.extract_features(normalized_split_value, 6))
        temp.append(test_data.anomaly)
        features.append(temp)
        print (feature_service.extract_features(normalized_split_value, 6))

print(features)

