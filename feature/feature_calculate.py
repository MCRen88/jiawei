


from feature.setting import ComprehensiveFCParameters,ComprehensiveFCParameters_feature_anom\
    , ComprehensiveFCParameters_feature_pattern,ComprehensiveFCParameters_feature_stat
import warnings
import time_series_detector.feature.feature_calculators_withoutparam as ts_feature_calculators_without_param
import feature.feature_anom as feature_anom
import pandas as pd

#############################################--s_features_with_parameter1--##########################################################

def _do_extraction_on_chunk(x, default_fc_parameters = None):
    """
    :param chunk: A tuple of sample_id, kind, data
    :param default_fc_parameters: A dictionary of feature calculators.
    :param kind_to_fc_parameters: A dictionary of fc_parameters for special kinds or None.
    :return: A list of calculated features.
    """
    data = x
    fc_paxrameters = ComprehensiveFCParameters()

    def _f():
        for function_name, parameter_list in fc_paxrameters.items():
            func = getattr(ts_feature_calculators_without_param, function_name)

            # If the function uses the index, pass is at as a pandas Series.
            # Otherwise, convert to numpy array
            if getattr(func, 'input', False) == 'pd.Series':
                # If it has a required index type, check that the data has the right index type.
                index_type = getattr(func, 'index_type', None)

                if index_type is not None:
                    try:
                        assert isinstance(data.index, index_type)
                    except AssertionError:
                        warnings.warn(
                            "{} requires the data to have a index of type {}. Results will "
                            "not be calculated".format(function_name, index_type)
                        )
                        continue
                a = data
            else:
                a = data

            if func.fctype == "combiner":
                result = func(a, param=parameter_list)
                # result = a
            #
            else:
                if parameter_list:
                    result = (func(a, **param) for param in
                              parameter_list)
                else:
                    result = [("", func(a))]


            for key, item in enumerate(result):
                feature_name = func.__name__
                if key:
                    feature_name += "__" + str(key)
                yield {"{}".format(feature_name):item}

    return list(_f())

def get_parameters_features(x, default_fc_parameters=None):
    """

    :param x:
    :param default_fc_parameters:
    :return:
    """
    if default_fc_parameters is None:
        default_fc_parameters = ComprehensiveFCParameters()
    k = _do_extraction_on_chunk(x)
    return k


#############################################--features_anom--##########################################################



def _do_extraction_on_chunk_test(x, default_fc_parameters = None, kind_to_fc_parameters = None):
    """
    :param chunk: A tuple of sample_id, kind, data
    :param default_fc_parameters: A dictionary of feature calculators.
    :param kind_to_fc_parameters: A dictionary of fc_parameters for special kinds or None.
    :return: A list of calculated features.
    """
    data = x
    fc_parameters = ComprehensiveFCParameters_feature_anom()
    # if kind_to_fc_parameters and kind in kind_to_fc_parameters:
    #     fc_parameters = kind_to_fc_parameters[kind]
    # else:
    #     fc_parameters = default_fc_parameters

    def _f():
        for function_name, parameter_list in fc_parameters.items():
            func = getattr(feature_anom, function_name)

            # If the function uses the index, pass is at as a pandas Series.
            # Otherwise, convert to numpy array
            if getattr(func, 'input', False) == 'pd.Series':
                # If it has a required index type, check that the data has the right index type.
                index_type = getattr(func, 'index_type', None)

                if index_type is not None:
                    try:
                        assert isinstance(data.index, index_type)
                    except AssertionError:
                        warnings.warn(
                            "{} requires the data to have a index of type {}. Results will "
                            "not be calculated".format(function_name, index_type)
                        )
                        continue
                a = data

            else:
                a = data

            if func.fctype == "combiner":
                result = func(a, param=parameter_list)
                # result = a
            #
            elif func.fctype == "simple":
                if parameter_list:
                    result = (func(a, **param) for param in
                              parameter_list)
                else:
                    result = [(func(a))]
            elif func.fctype == "binned":
                if parameter_list:
                    result = (func(a, **param) for param in
                              parameter_list)
                else:
                    result = (func(a))
            elif func.fctype == "test":
                if parameter_list:
                    result = (func(a, **param) for param in
                              parameter_list)
                else:
                    result = func(a)


            for key, item in enumerate(result):
                feature_name = func.__name__
                if key:
                    feature_name += "__" + str(key)
                yield {"{}".format(feature_name):item}

    return list(_f())

def get_classification_features_test(x, default_fc_parameters=None,kind_to_fc_parameters = None):
    """

    :param x:
    :param default_fc_parameters:
    :return:
    """
    ####     for feature_anom.py
    k = _do_extraction_on_chunk_test(x)
    return k

#############################################--features_pattern--##########################################################


def _do_extraction_on_chunk_feature_pattern(x, default_fc_parameters = None, kind_to_fc_parameters = None):
    """
    :param chunk: A tuple of sample_id, kind, data
    :param default_fc_parameters: A dictionary of feature calculators.
    :param kind_to_fc_parameters: A dictionary of fc_parameters for special kinds or None.
    :return: A list of calculated features.
    """
    data = x
    fc_parameters = ComprehensiveFCParameters_feature_pattern()
    # if kind_to_fc_parameters and kind in kind_to_fc_parameters:
    #     fc_parameters = kind_to_fc_parameters[kind]
    # else:
    #     fc_parameters = default_fc_parameters

    def _f():
        for function_name, parameter_list in fc_parameters.items():
            func = getattr(feature_anom, function_name)

            # If the function uses the index, pass is at as a pandas Series.
            # Otherwise, convert to numpy array
            if getattr(func, 'input', False) == 'pd.Series':
                # If it has a required index type, check that the data has the right index type.
                index_type = getattr(func, 'index_type', None)

                if index_type is not None:
                    try:
                        assert isinstance(data.index, index_type)
                    except AssertionError:
                        warnings.warn(
                            "{} requires the data to have a index of type {}. Results will "
                            "not be calculated".format(function_name, index_type)
                        )
                        continue
                a = data

            else:
                a = data

            if func.fctype == "combiner":
                result = func(a, param=parameter_list)
                # result = a
            #
            elif func.fctype == "simple":
                if parameter_list:
                    result = (func(a, **param) for param in
                              parameter_list)
                else:
                    result = [(func(a))]
            elif func.fctype == "binned":
                if parameter_list:
                    result = (func(a, **param) for param in
                              parameter_list)
                else:
                    result = (func(a))
            elif func.fctype == "test":
                if parameter_list:
                    result = (func(a, **param) for param in
                              parameter_list)
                else:
                    result = func(a)


            for key, item in enumerate(result):
                feature_name = func.__name__
                if key:
                    feature_name += "__" + str(key)
                yield {"{}".format(feature_name):item}

    return list(_f())

def get_classification_feature_pattern(x, default_fc_parameters=None,kind_to_fc_parameters = None):
    """

    :param x:
    :param default_fc_parameters:
    :return:
    """
    ####     for feature_anom.py

    k = _do_extraction_on_chunk_feature_pattern(x)
    return k

#############################################--features_stat--##########################################################


def _do_extraction_on_chunk_feature_stat(x, default_fc_parameters = None, kind_to_fc_parameters = None):
    """
    :param chunk: A tuple of sample_id, kind, data
    :param default_fc_parameters: A dictionary of feature calculators.
    :param kind_to_fc_parameters: A dictionary of fc_parameters for special kinds or None.
    :return: A list of calculated features.
    """
    data = x
    fc_parameters = ComprehensiveFCParameters_feature_stat()
    # if kind_to_fc_parameters and kind in kind_to_fc_parameters:
    #     fc_parameters = kind_to_fc_parameters[kind]
    # else:
    #     fc_parameters = default_fc_parameters

    def _f():
        for function_name, parameter_list in fc_parameters.items():
            func = getattr(feature_anom, function_name)

            # If the function uses the index, pass is at as a pandas Series.
            # Otherwise, convert to numpy array
            if getattr(func, 'input', False) == 'pd.Series':
                # If it has a required index type, check that the data has the right index type.
                index_type = getattr(func, 'index_type', None)

                if index_type is not None:
                    try:
                        assert isinstance(data.index, index_type)
                    except AssertionError:
                        warnings.warn(
                            "{} requires the data to have a index of type {}. Results will "
                            "not be calculated".format(function_name, index_type)
                        )
                        continue
                a = data

            else:
                a = data

            if func.fctype == "combiner":
                result = func(a, param=parameter_list)
                # result = a
            #
            elif func.fctype == "simple":
                if parameter_list:
                    result = (func(a, **param) for param in
                              parameter_list)
                else:
                    result = [(func(a))]
            elif func.fctype == "binned":
                if parameter_list:
                    result = (func(a, **param) for param in
                              parameter_list)
                else:
                    result = (func(a))
            elif func.fctype == "test":
                if parameter_list:
                    result = (func(a, **param) for param in
                              parameter_list)
                else:
                    result = func(a)


            for key, item in enumerate(result):
                feature_name = func.__name__
                if key:
                    feature_name += "__" + str(key)
                yield {"{}".format(feature_name):item}

    return list(_f())

def get_classification_feature_pattern(x, default_fc_parameters=None,kind_to_fc_parameters = None):
    """

    :param x:
    :param default_fc_parameters:
    :return:
    """
    ####     for feature_anom.py

    k = _do_extraction_on_chunk_feature_stat(x)
    return k