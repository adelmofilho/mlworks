def exec_impute_missing(data, plan, key):

    features = plan[key]

    if key == "impute_missing_as_inf":

        data[features] = data[features].fillna(float('Inf'))

    elif key == "impute_missing_as_category":

        data[features] = data[features].fillna('missing')

    elif key == "impute_missing_as_zero":

        data[features] = data[features].fillna(0)

    elif key == "impute_missing_as_number":

        for var in features:

            data[var] = data[var].fillna(features[var])

    return data


def exec_binning_one_hot_encoding(data, plan):
    pass
