from datasets.filter.Selector import Selector
from datasets.filter.SortByImageRatio import SortByImageRatio
from datasets.filter.DataCleaning.Integrity import DataCleaning_Integrity
from datasets.filter.DataCleaning.BoundingBox import DataCleaning_BoundingBox
from datasets.filter.DataCleaning.AnnotationStandard import DataCleaning_AnnotationStandard
from .tweak_tool import ImageDatasetTweakTool
from datasets.filter.DataCleaning.ObjectCategory import DataCleaning_ObjectCategory


__all__ = ['apply_filters_on_image_dataset_']


def apply_filters_on_image_dataset_(dataset: dict, filters: list):
    if len(filters) == 0:
        return dataset

    if 'filters' not in dataset:
        dataset['filters'] = []

    filters_backup = dataset['filters']

    dataset_tweak_tool = ImageDatasetTweakTool(dataset)

    for filter_ in filters:
        if isinstance(filter_, Selector):
            dataset_tweak_tool.apply_index_filter(filter_(len(dataset['sequences'])))
        elif isinstance(filter_, SortByImageRatio):
            dataset_tweak_tool.sort_by_image_size_ratio(filter_.descending)
        elif isinstance(filter_, DataCleaning_BoundingBox):
            if filter_.fit_in_image_size:
                dataset_tweak_tool.bounding_box_fit_in_image_size()
            if filter_.update_validity:
                dataset_tweak_tool.bounding_box_update_validity()
            if filter_.remove_invalid_objects:
                dataset_tweak_tool.bounding_box_remove_non_validity_objects()
            if filter_.remove_empty_objects:
                dataset_tweak_tool.bounding_box_remove_empty_annotation_objects()
        elif isinstance(filter_, DataCleaning_Integrity):
            if filter_.remove_zero_annotation_image:
                dataset_tweak_tool.remove_empty_annotation()
            if filter_.remove_invalid_image:
                dataset_tweak_tool.remove_invalid_image()
        elif isinstance(filter_, DataCleaning_ObjectCategory):
            if filter_.category_ids_to_remove is not None:
                dataset_tweak_tool.remove_category_ids(filter_.category_ids_to_remove)
            if filter_.make_category_id_sequential:
                dataset_tweak_tool.make_category_id_sequential()
        elif isinstance(filter_, DataCleaning_AnnotationStandard):
            dataset_tweak_tool.annotation_standard_conversion(filter_.bounding_box_format,
                                                              filter_.pixel_coordinate_system,
                                                              filter_.bounding_box_coordinate_system,
                                                              filter_.pixel_definition)
        else:
            raise RuntimeError(f"{type(filter_)} not implemented for Image Dataset")

        filters_backup.append(filter_.serialize())
    dataset['filters'] = filters_backup
