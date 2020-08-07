def exec_impute_missing_as_category(df, plan):

    features = plan["impute_missing_as_category"]

    df[features] = df[features].fillna('missing', inplace=True)

    return df
        
