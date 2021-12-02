import os
from datasets.types.data_split import DataSplit
from data.types.bounding_box_format import BoundingBoxFormat
from data.types.pixel_coordinate_system import PixelCoordinateSystem
from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
from data.types.pixel_definition import PixelDefinition
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import datasets.base.video.dataset
import datasets.base.image.dataset

image_dataset_key_exclude_list = ('name', 'split', 'version', 'filters', 'type', 'category_id_name_map', 'images', 'context')
image_dataset_image_key_exclude_list = ('size', 'path', 'objects', 'category_id')
image_dataset_object_key_exclude_list = ('category_id', 'bounding_box')

video_dataset_key_exclude_list = ('name', 'split', 'version', 'filters', 'type', 'category_id_name_map', 'sequences', 'context')
video_dataset_sequence_key_exclude_list = ('name', 'path', 'fps', 'frames', 'objects')
video_dataset_frame_key_exclude_list = ('path', 'size', 'objects')
video_dataset_sequence_object_key_exclude_list = ('category_id', 'id')
video_dataset_frame_object_key_exclude_list = ('id', 'bounding_box')


class DatasetProcessBar:
    def __init__(self):
        self.pbar = None
        self.total = None
        self.dataset_name = None
        self.dataset_split = None
        self.sequence_name = None

    def _construct_bar_if_not_exists(self):
        if self.pbar is None:
            self.pbar = tqdm(total=self.total)
            self._update_pbar_desc()

    def set_total(self, total: int):
        assert isinstance(total, int)
        self.total = total

    def _update_pbar_desc(self):
        if self.pbar is None:
            return
        assert self.dataset_name is not None
        string = self.dataset_name
        if self.dataset_split is not None:
            string += f'({self.dataset_split})'
        self.pbar.set_description_str(string)
        if self.sequence_name is not None:
            self.pbar.set_postfix_str(self.sequence_name)

    def set_dataset_name(self, name: str):
        self.dataset_name = name
        self._update_pbar_desc()

    def set_dataset_split(self, split: DataSplit):
        if split == DataSplit.Full:
            self.dataset_split = None
            return
        self.dataset_split = split.name
        self._update_pbar_desc()

    def set_sequence_name(self, name: str):
        self.sequence_name = name
        self._update_pbar_desc()

    def update(self, n=1):
        self._construct_bar_if_not_exists()
        self.pbar.update(n)

    def close(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class _DatasetConstructionContext:
    def __init__(self):
        self.bounding_box_data_type = None
        self.bounding_box_format = None
        self.pixel_coordinate_system = None
        self.bounding_box_coordinate_system = None
        self.pixel_definition = None
        self.pbar = DatasetProcessBar()

    def initialize_from(self, dataset: dict):
        if 'context' in dataset:
            dataset_context = dataset['context']
            if 'bounding_box_data_type' in dataset_context:
                if dataset_context['bounding_box_data_type'] == 'int':
                    self.bounding_box_data_type = int
                elif dataset_context['bounding_box_data_type'] == 'float':
                    self.bounding_box_data_type = float
                else:
                    raise RuntimeError(f"unknown value {dataset_context['bounding_box_data_type']} in dataset['bounding_box_data_type']")
            if 'bounding_box_format' in dataset_context:
                self.bounding_box_format = BoundingBoxFormat[dataset_context['bounding_box_format']]
            if 'pixel_coordinate_system' in dataset_context:
                self.pixel_coordinate_system = PixelCoordinateSystem[dataset_context['pixel_coordinate_system']]
            if 'pixel_definition' in dataset_context:
                self.pixel_definition = PixelDefinition[dataset_context['pixel_definition']]
            if 'bounding_box_coordinate_system' in dataset_context:
                self.bounding_box_coordinate_system = BoundingBoxCoordinateSystem[dataset_context['bounding_box_coordinate_system']]

    def dump_with_default_value(self, dataset: dict):
        if self.bounding_box_data_type is None:
            return

        if 'context' not in dataset:
            dataset['context'] = {}
        dataset_context = dataset['context']

        if self.bounding_box_data_type == int:
            dataset_context['bounding_box_data_type'] = 'int'
        elif self.bounding_box_data_type == float:
            dataset_context['bounding_box_data_type'] = 'float'
        elif self.bounding_box_data_type == 'mixed':
            dataset_context['bounding_box_data_type'] = 'float'
        else:
            raise RuntimeError(f'Unknown value {self.bounding_box_data_type}')

        if self.bounding_box_format is None:
            dataset_context['bounding_box_format'] = BoundingBoxFormat.XYWH.name
        else:
            dataset_context['bounding_box_format'] = self.bounding_box_format.name

        if self.bounding_box_coordinate_system is None:
            if dataset_context['bounding_box_data_type'] == 'float':
                dataset_context['bounding_box_coordinate_system'] = BoundingBoxCoordinateSystem.Spatial.name
            else:
                dataset_context['bounding_box_coordinate_system'] = BoundingBoxCoordinateSystem.Rasterized.name
        else:
            dataset_context['bounding_box_coordinate_system'] = self.bounding_box_coordinate_system.name

        if self.pixel_coordinate_system is None:
            if dataset_context['bounding_box_data_type'] == 'float':
                dataset_context['pixel_coordinate_system'] = PixelCoordinateSystem.HalfPixelOffset.name
            else:
                dataset_context['pixel_coordinate_system'] = PixelCoordinateSystem.Aligned.name
        else:
            dataset_context['pixel_coordinate_system'] = self.pixel_coordinate_system.name

        if self.pixel_definition is None:
            dataset_context['pixel_definition'] = PixelDefinition.Point.name
        else:
            dataset_context['pixel_definition'] = self.pixel_definition.name

    def set_bounding_box_dtype(self, type_):
        if self.bounding_box_data_type is None:
            self.bounding_box_data_type = type_
            return
        if self.bounding_box_data_type != type_:
            self.bounding_box_data_type = 'mixed'

    def set_bounding_box_format(self, format_):
        assert isinstance(format_, BoundingBoxFormat)
        self.bounding_box_format = format_

    def get_processing_bar(self):
        return self.pbar

    def set_pixel_coordinate_system(self, pixel_coordinate_system: PixelCoordinateSystem):
        assert isinstance(pixel_coordinate_system, PixelCoordinateSystem)
        self.pixel_coordinate_system = pixel_coordinate_system

    def set_bounding_box_coordinate_system(self, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
        assert isinstance(bounding_box_coordinate_system, BoundingBoxCoordinateSystem)
        self.bounding_box_coordinate_system = bounding_box_coordinate_system

    def set_pixel_definition(self, pixel_definition: PixelDefinition):
        assert isinstance(pixel_definition, PixelDefinition)
        self.pixel_definition = pixel_definition

    def __del__(self):
        self.pbar.close()


def _root_path_impl(root_path):
    return os.path.abspath(root_path)


def _add_path_impl(path, root_path):
    path = os.path.abspath(path)
    rel_path = os.path.relpath(path, root_path)
    return rel_path


def set_path_(image_dict: dict, image_path: str, root_path: str, size):
    image_dict['path'] = _add_path_impl(image_path, root_path)
    if size is None:
        try:
            image = Image.open(image_path)
            image_dict['size'] = image.size
        except UnidentifiedImageError:
            print(f'Warning: failed to decode image file {image_path}')
            image_dict['size'] = (0, 0)
    else:
        assert len(size) == 2
        for v in size:
            if isinstance(v, float):
                assert v.is_integer()
        size = tuple([int(v) for v in size])
        image_dict['size'] = size


def generate_sequence_path_(sequence: dict):
    assert 'path' not in sequence
    paths = []
    for frame in sequence['frames']:
        paths.append(frame['path'])
    sequence_path = os.path.commonpath(paths)
    sequence['path'] = sequence_path
    for frame in sequence['frames']:
        frame['path'] = os.path.relpath(frame['path'], sequence_path)


class BaseDatasetSequenceConstructor:
    def __init__(self, sequence: dict, root_path: str, context: _DatasetConstructionContext):
        self.sequence = sequence
        self.root_path = root_path
        self.context = context

    def set_name(self, name: str):
        self.sequence['name'] = name
        self.context.get_processing_bar().set_sequence_name(name)

    def set_fps(self, fps):
        self.sequence['fps'] = fps

    def set_attribute(self, name, value):
        self.sequence[name] = value

    def merge_attributes(self, attributes):
        for key, value in attributes.items():
            self.sequence[key] = value


class BaseDatasetImageConstructor:
    def __init__(self, image: dict, root_path: str, context: _DatasetConstructionContext, category_id_name_map: dict=None):
        self.image = image
        self.root_path = root_path
        self.context = context
        self.category_id_name_map = category_id_name_map

    def set_category_id(self, category_id):
        assert self.category_id_name_map is not None, "set set_category_id_name_map first"
        assert category_id in self.category_id_name_map
        self.image['category_id'] = category_id

    def set_attribute(self, name: str, value):
        self.image[name] = value

    def set_path(self, path: str, image_size=None):
        set_path_(self.image, path, self.root_path, image_size)

    def get_image_size(self):
        return self.image['size']

    def merge_attributes(self, attributes):
        for key, value in attributes.items():
            self.image[key] = value


class BaseDatasetImageConstructorGenerator:
    def __init__(self, context: _DatasetConstructionContext):
        self.context = context

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context.get_processing_bar().update()


class BaseDatasetSequenceConstructorGenerator:
    def __init__(self, sequence: dict, context: _DatasetConstructionContext):
        self.sequence = sequence
        self.context = context

    def __exit__(self, exc_type, exc_val, exc_tb):
        generate_sequence_path_(self.sequence)
        self.context.get_processing_bar().update()


class _BaseDatasetConstructor:
    def __init__(self, dataset: dict, root_path: str, dataset_type: str, schema_version: int, data_version: int, context: _DatasetConstructionContext):
        if 'type' in dataset:
            assert dataset['type'] == dataset_type
        else:
            dataset['type'] = dataset_type
        if 'version' in dataset:
            assert dataset['version'][0] == schema_version
            assert dataset['version'][1] == data_version
        else:
            dataset['version'] = [schema_version, data_version]

        assert 'filters' not in dataset or len(dataset['filters']) == 0

        self.dataset = dataset
        self.root_path = _root_path_impl(root_path)
        self.context = context

    def set_category_id_name_map(self, category_id_name_map: dict):
        self.dataset['category_id_name_map'] = category_id_name_map

    def set_name(self, name: str):
        self.dataset['name'] = name
        self.context.get_processing_bar().set_dataset_name(name)

    def set_split(self, split: DataSplit):
        assert len(split.name) > 0
        self.dataset['split'] = split.name
        self.context.get_processing_bar().set_dataset_split(split)

    def set_pixel_coordinate_system(self, pixel_coordinate_system: PixelCoordinateSystem):
        self.context.set_pixel_coordinate_system(pixel_coordinate_system)

    def set_bounding_box_format(self, bounding_box_format: BoundingBoxFormat):
        self.context.set_bounding_box_format(bounding_box_format)

    def set_bounding_box_coordinate_system(self, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
        self.context.set_bounding_box_coordinate_system(bounding_box_coordinate_system)

    def set_pixel_definition(self, pixel_definition: PixelDefinition):
        self.context.set_pixel_definition(pixel_definition)


class BaseVideoDatasetConstructor(_BaseDatasetConstructor):
    def __init__(self, dataset: dict, root_path: str, data_version: int, context: _DatasetConstructionContext):
        super(BaseVideoDatasetConstructor, self).__init__(dataset, root_path, 'video', datasets.base.video.dataset.__version__, data_version, context)

    def set_total_number_of_sequences(self, number: int):
        self.context.get_processing_bar().set_total(number)

    def set_attribute(self, name: str, value):
        self.dataset[name] = value


class BaseImageDatasetConstructor(_BaseDatasetConstructor):
    def __init__(self, dataset: dict, root_path: str, data_version: int, context: _DatasetConstructionContext):
        super(BaseImageDatasetConstructor, self).__init__(dataset, root_path, 'image', datasets.base.image.dataset.__version__, data_version, context)
        if 'images' not in dataset:
            dataset['images'] = []

    def set_total_number_of_images(self, number: int):
        self.context.get_processing_bar().set_total(number)

    def set_attribute(self, name: str, value):
        self.dataset[name] = value


class BaseDatasetConstructorGenerator:
    def __init__(self, dataset: dict, root_path: str, version: int, constructor_type):
        self.dataset = dataset
        self.root_path = root_path
        self.version = version
        self.constructor_type = constructor_type

    def __enter__(self):
        self.context = _DatasetConstructionContext()
        self.context.initialize_from(self.dataset)
        return self.constructor_type(self.dataset, self.root_path, self.version, self.context)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert 'split' in self.dataset
        assert 'name' in self.dataset

        self.context.dump_with_default_value(self.dataset)

        del self.context
