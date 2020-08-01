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
            plan[key]["var"] = features
            plan[key]["value"] = value
        else:
            plan[key]["var"] = plan[key]["var"] + features
            plan[key]["value"] = plan[key]["value"] + value
        return plan

    def __include_on_custom_list

    def __include_on_order_list(plan, key, feature, order):
        return None

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

    def transform_category_to_order(self, feature, order):
        key = "transform_category_to_order"
        self.plan = self.__include_on_order_list(self.plan, key, feature, order)
        return self


    def transform_linear(self, feature, operation):
        key = "transform_linear"
        self.plan = self.__include_on_operation_list(self.plan, key, feature, operation)
        return self