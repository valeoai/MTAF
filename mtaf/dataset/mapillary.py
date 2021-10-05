import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils import data
from mtaf.utils import project_root
from mtaf.utils.serialization import json_load

# from valeodata import download

DEFAULT_INFO_PATH = project_root / 'mtaf/dataset/mapillary_list/info.json'

class MapillaryDataSet(data.Dataset):

    classes_ids = {'flat': 0,
                   'construction': 1,
                   'object': 2,
                   'nature': 3,
                   'sky': 4,
                   'human': 5,
                   'vehicle': 6,
                   'other': 255}

    classes_mappings_mapillary_to_cityscapes = {'bird': 'other',
                                                'ground animal': 'other',
                                                'curb': 'construction',
                                                'fence': 'construction',
                                                'guard rail': 'construction',
                                                'barrier': 'construction',
                                                'wall': 'construction',
                                                'bike lane': 'flat',
                                                'crosswalk - plain': 'flat',
                                                'curb cut': 'flat',
                                                'parking': 'flat',
                                                'pedestrian area': 'flat',
                                                'rail track': 'flat',
                                                'road': 'flat',
                                                'service lane': 'flat',
                                                'sidewalk': 'flat',
                                                'bridge': 'construction',
                                                'building': 'construction',
                                                'tunnel': 'construction',
                                                'person': 'human',
                                                'bicyclist': 'human',
                                                'motorcyclist': 'human',
                                                'other rider': 'human',
                                                'lane marking - crosswalk': 'flat',
                                                'lane marking - general': 'flat',
                                                'mountain': 'other',
                                                'sand': 'other',
                                                'sky': 'sky',
                                                'snow': 'other',
                                                'terrain': 'flat',
                                                'vegetation': 'nature',
                                                'water': 'other',
                                                'banner': 'other',
                                                'bench': 'other',
                                                'bike rack': 'other',
                                                'billboard': 'other',
                                                'catch basin': 'other',
                                                'cctv camera': 'other',
                                                'fire hydrant': 'other',
                                                'junction box': 'other',
                                                'mailbox': 'other',
                                                'manhole': 'other',
                                                'phone booth': 'other',
                                                'pothole': 'object',
                                                'street light': 'object',
                                                'pole': 'object',
                                                'traffic sign frame': 'object',
                                                'utility pole': 'object',
                                                'traffic light': 'object',
                                                'traffic sign (back)': 'object',
                                                'traffic sign (front)': 'object',
                                                'trash can': 'other',
                                                'bicycle': 'vehicle',
                                                'boat': 'vehicle',
                                                'bus': 'vehicle',
                                                'car': 'vehicle',
                                                'caravan': 'vehicle',
                                                'motorcycle': 'vehicle',
                                                'on rails': 'vehicle',
                                                'other vehicle': 'vehicle',
                                                'trailer': 'vehicle',
                                                'truck': 'vehicle',
                                                'wheeled slow': 'vehicle',
                                                'car mount': 'other',
                                                'ego vehicle': 'other',
                                                'unlabeled': 'other'}

    def __init__(self, root, set='train', max_iters=None, crop_size=(321, 321), mean=(128, 128, 128),
                 class_mappings=classes_mappings_mapillary_to_cityscapes, model_classes=classes_ids,
                 load_instances=False, scale_label=True, labels_size=None, info_path=DEFAULT_INFO_PATH):
        # root = Path(download('mapillary'))
        self.path = Path(root) / set
        self.crop_size = crop_size
        if labels_size is None:
            self.labels_size = crop_size
        else:
            self.labels_size = labels_size
        self.load_instances = load_instances
        self.mean = mean
        self.scale_label = scale_label
        self.info = json_load(info_path)

        sorted_paths = map(lambda x: sorted((self.path / x).iterdir()),
                           ('images', 'labels', 'instances'))

        self.data_paths = list(zip(*sorted_paths))
        if max_iters is not None:
            self.data_paths = self.data_paths * int(np.ceil(float(max_iters) / len(self.data_paths)))

        self.labels = json.loads((Path(root) / 'config.json').read_text())['labels']

        self.vector_mappings = None
        if class_mappings is not None:
            dataset_classes = [label['readable'] for label in self.labels]
            self.vector_mappings = array_from_class_mappings(dataset_classes,
                                                             class_mappings,
                                                             model_classes)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path, labels_path, instances_path = self.data_paths[index]

        image_array = Image.open(image_path)  # 3D  #double-check if this is RGB
        image_array = resize_with_pad(self.crop_size, image_array, Image.BICUBIC)
        size = image_array.shape
        image_array = (image_array[:, :, ::-1] - self.mean).transpose((2, 0, 1))
        labels_array = Image.open(labels_path)  # 2D
        if self.scale_label:
            labels_array = resize_with_pad(self.labels_size, labels_array,
                                           Image.NEAREST,
                                           fill_value=len(self.labels) - 1)
        else:
            labels_array = pad_with_fixed_AS(self.labels_size[0]/self.labels_size[1], labels_array,
                                           fill_value=len(self.labels) - 1)
        if self.load_instances:
            warnings.warn('The code for the instances is unfinished.')
            instances_array = np.array(Image.open(instances_path))  # 2D

        if self.vector_mappings is not None:
            # we have to remap the labels on new classes.
            # labels_array = self.vector_mappings[labels_array]
            labels_array = label_mapping_mapilliary(labels_array, self.vector_mappings)

        if self.load_instances:
            return image_array, labels_array, instances_array
        else:
            return image_array, labels_array, np.array(size), str(image_path)


def label_mapping_mapilliary(input, mapping):
    output = np.copy(input)
    for ind,val in enumerate(mapping):
        output[input == ind] = val
    return np.array(output, dtype=np.int64)

def array_from_class_mappings(dataset_classes, class_mappings, model_classes):
    """
    :param dataset_classes: list or dict. Mapping between indexes and name of classes.
                            If using a list, it's equivalent
                            to {x: i for i, x in enumerate(dataset_classes)}
    :param class_mappings: Dictionary mapping names of the dataset to
                           names of classes of the model.
    :param model_classes:  list or dict. Same as dataset_classes,
                           but for the model classes.
    :return: A numpy array representing the mapping to be done.
    """
    # Assert all classes are different.
    assert len(model_classes) == len(set(model_classes))

    # to generate the template to fill the dictionary for class_mappings
    # uncomment this code.
    """
    for x in dataset_classes:
        print((' ' * 20) + f'\'{name}\': \'\',')
    """

    # Not case sensitive to make it easier to write.
    if isinstance(dataset_classes, list):
        dataset_classes = {x: i for i, x in enumerate(dataset_classes)}
    dataset_classes = {k.lower(): v for k, v in dataset_classes.items()}
    class_mappings = {k.lower(): v.lower() for k, v in class_mappings.items()}
    if isinstance(model_classes, list):
        model_classes = {x: i for i, x in enumerate(model_classes)}
    model_classes = {k.lower(): v for k, v in model_classes.items()}

    result = np.zeros((max(dataset_classes.values()) + 1,), dtype=np.uint8)
    for dataset_class_name, i in dataset_classes.items():
        result[i] = model_classes[class_mappings[dataset_class_name]]
    return result


def resize_with_pad(target_size, image, resize_type, fill_value=0):
    if target_size is None:
        return np.array(image)
    # find which size to fit to the target size
    target_ratio = target_size[0] / target_size[1]
    image_ratio = image.size[0] / image.size[1]

    if image_ratio > target_ratio:
        resize_ratio = target_size[0] / image.size[0]
        new_image_shape = (target_size[0], int(image.size[1] * resize_ratio))
    else:
        resize_ratio = target_size[1] / image.size[1]
        new_image_shape = (int(image.size[0] * resize_ratio), target_size[1])

    image_resized = image.resize(new_image_shape, resize_type)

    image_resized = np.array(image_resized)
    if image_resized.ndim == 2:
        image_resized = image_resized[:, :, None]
    tmp = target_size[::-1] + [image_resized.shape[2],]
    result = np.ones(tmp, image_resized.dtype) * fill_value
    assert image_resized.shape[0] <= result.shape[0]
    assert image_resized.shape[1] <= result.shape[1]
    placeholder = result[:image_resized.shape[0], :image_resized.shape[1]]
    placeholder[:] = image_resized
    return result

def pad_with_fixed_AS(target_ratio, image, fill_value=0):
    dimW = float(image.size[0])
    dimH = float(image.size[1])
    image_ratio = dimW/dimH
    if target_ratio > image_ratio:
        dimW = target_ratio*dimH
    elif target_ratio < image_ratio:
        dimH = dimW/target_ratio
    else:
        return np.array(image)
    image = np.array(image)
    result = np.ones((int(dimH), int(dimW)), image.dtype) * fill_value
    placeholder = result[:image.shape[0], :image.shape[1]]
    placeholder[:] = image
    return result
