import copy


class DatasetCategoryIdNameMapAutoGeneratorTool_FromCategoryNames:
    def __init__(self):
        self.category_names = []

    def add_category_name(self, name):
        if name not in self.category_names:
            self.category_names.append(name)

    def generate(self):
        return DatasetCategoryIdNameMapAutoGeneratorTool_FromCategoryNames.generate_from_list(self.category_names)

    @staticmethod
    def generate_from_list(category_names):
        category_names = copy.copy(category_names)
        category_names.sort()
        return {k: v for k, v in enumerate(category_names)}
