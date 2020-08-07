def exec_impute_missing(data, plan, key):

    features = plan[key]

    if key == "impute_missing_as_inf":

        data[features] = data[features].fillna(float('Inf'))

    elif key == "impute_missing_as_category":

        data[features] = data[features].fillna('missing')

    return data
