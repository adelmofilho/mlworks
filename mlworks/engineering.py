from mlworks.executors import exec_impute_missing


class Blueprint:

    def __init__(self, data):
        self.data = data

    def create_plan(self):
        self.plan = dict()

    def __include_on_list(self, plan, key, features):

        if plan.get(key, True) is True:
            plan[key] = features
        else:
            plan[key] = plan[key] + features
        return plan

    def __include_on_pairwise_list(self, plan, key, features, value):

        if plan.get(key, True) is True:
            plan[key] = {}
            if len(value) == 1:
                value = value * len(features)
        for var_index in range(len(features)):
            plan[key][features[var_index]] = features[var_index]
            plan[key][features[var_index]] = value[var_index]
        return plan

    def __include_on_custom_list(self, plan, key, feature, classes):

        if plan.get(key, True) is True:
            plan[key] = {}
        plan[key][feature] = {}
        for var_index in range(len(classes)):
            plan[key][feature]["class" + str(var_index)] = classes[var_index]
        return plan

    def __include_on_order_list(self, plan, key, features, order, grade=None):

        if plan.get(key, True) is True:
            plan[key] = {}
        for var_index in range(len(features)):
            plan[key][features[var_index]] = {}
            plan[key][features[var_index]]["order"] = order[var_index]
            if grade is None:
                gradeOrder = list(reversed(list(range(len(order[var_index])))))
            else:
                gradeOrder = grade[var_index]
            plan[key][features[var_index]]["grade"] = gradeOrder
        return plan

    def __include_on_operation_list(self, key, feature, operation):
        return None

    # Original Feature

    def keep_original_feature(self, features):
        key = "keep_original_feature"
        self.plan = self.__include_on_list(self.plan, key, features)
        return self

    # Imputation methods

    def impute_missing_as_category(self, features):
        key = "impute_missing_as_category"
        self.plan = self.__include_on_list(self.plan, key, features)
        return self

    def impute_missing_as_inf(self, features):
        key = "impute_missing_as_inf"
        self.plan = self.__include_on_list(self.plan, key, features)
        return self

    def impute_missing_as_zero(self, features):
        key = "impute_missing_as_zero"
        self.plan = self.__include_on_list(self.plan, key, features)
        return self

    def impute_missing_as_number(self, features, number):
        key = "impute_missing_as_number"
        self.plan = self.__include_on_pairwise_list(self.plan, key, features, number)
        return self

    # Binning

    def binning_number_one_threshold(self, features, thresholds):
        key = "binning_number_one_threshold"
        self.plan = self.__include_on_pairwise_list(self.plan, key, features, thresholds)
        return self

    def binning_class_one_vs_all(self, features, hot_class):
        key = "binning_class_one_vs_all"
        self.plan = self.__include_on_pairwise_list(self.plan, key, features, hot_class)
        return self

    def binning_one_hot_encoding(self, features):
        key = "binning_one_hot_encoding"
        self.plan = self.__include_on_list(self.plan, key, features)
        return self

    def binning_add_extra_class(self, features, extra_class):
        key = "binning_add_extra_class"
        self.plan = self.__include_on_pairwise_list(self.plan, key, features, extra_class)
        return self

    def binning_custom_classes(self, features, extra_class):
        key = "binning_custom_classes"
        self.plan = self.__include_on_custom_list(self.plan, key, features, extra_class)
        return self

    # Transform

    def transform_category_to_order(self, features, order, grade=None):
        key = "transform_category_to_order"
        self.plan = self.__include_on_order_list(self.plan, key, features, order, grade=None)
        return self

    def transform_linear(self, features, number):
        key = "transform_linear"
        self.plan = self.__include_on_pairwise_list(self.plan, key, features, number)
        return self

    # Method Execute

    def execute(self):

        self.eng_data = self.data

        self.eng_data = exec_impute_missing(
            data=self.eng_data,
            plan=self.plan,
            key="impute_missing_as_inf")

        self.eng_data = exec_impute_missing(
            data=self.eng_data,
            plan=self.plan,
            key="impute_missing_as_category")

        self.eng_data = exec_impute_missing(
            data=self.eng_data,
            plan=self.plan,
            key="impute_missing_as_zero")

        self.eng_data = exec_impute_missing(
            data=self.eng_data,
            plan=self.plan,
            key="impute_missing_as_number")

        return self.eng_data

#         X_columns = []
#         # #Dummy faltante
#         for var in self.params["insert_dummy_faltante"].keys():
#             dummy_faltante = pd.get_dummies(data=df[var], drop_first = True)
#             if self.params["insert_dummy_faltante"][var]["dummy"] in dummy_faltante.columns:
#                 a = 1
#             else:
#                 dummy_faltante[self.params["insert_dummy_faltante"][var]["dummy"]] = 0
#             df[list(dummy_faltante.columns)] = dummy_faltante
#             X_columns =  X_columns + list(dummy_faltante.columns)
#         # #Dummy controlada
#         for var in self.params["dummy_controlada"].keys():
#             for classe in list(self.params["dummy_controlada"][var].keys()):
#                 new_var = str(var) + "_" + str(classe)
#                 X_columns = X_columns + [new_var]
#                 df[new_var] = df[var].fillna("missing").isin(self.params["dummy_controlada"][var][classe])/
#         # # Missing number to Inf
#         df[self.params["missing_number_to_inf"]] = df[self.params["missing_number_to_inf"]].fillna('missing')
#         for var in self.params["missing_number_to_inf"]:
#             X_columns = X_columns +  [var + '_miss_num']
#             df[[var + '_miss_num']] = df[[var]].replace("missing", -100 )
#         # # Binary to class
#         binary_dummy = pd.get_dummies(data=df[self.params["binary_dummies"]], drop_first = True)
#         df[list(binary_dummy.columns)] = binary_dummy
#         # X_columns =  X_columns + list(binary_dummy.columns)
#         # Unify to class
#         for var in self.params["unify_classes"].keys():
#             X_columns = X_columns +  [var + '_unify']
#             class1 = self.params["unify_classes"][var]["class"]
#             df.loc[df[var] != class1, var] = 0
#             df.loc[df[var] == class1, var] = 1
#             df[[var + '_unify']] = df[[var]]
#         # # Alteração de escala
#         for var in self.params["scale_adjust"].keys():
#             X_columns = X_columns +  [var + '_adj']
#             df[[var + '_adj']] = df[[var]] + self.params["scale_adjust"][var]["value"]
#         # # continuous to binary
#         for var in self.params["continuous_to_binary"].keys():
#             X_columns = X_columns +  [var + '_binary']
#             df[[var + '_binary']] = (df[[var]] > self.params["continuous_to_binary"][var]["threshold"]).astype(int)
#         # # Factor to number
#         df[list(self.params["factor_to_number"].keys())] = df[list(self.params["factor_to_number"].keys())].fillna('missing')
#         for var in self.params["factor_to_number"].keys():
#             X_columns = X_columns +  [var + '_grade']
#             df[[var + '_grade']] = df[[var]].\
#                 replace(self.params["factor_to_number"][var]["order"],
#                         self.params["factor_to_number"][var]["grade"])
#         # # Selected Variables
#         X_columns = X_columns + self.params["identity"] + list(binary_dummy.columns)
#         # feature_table = df[set(X_columns)]
#         # return feature_table
        # return df[X_columns]
        # return pass
