import numpy as np

from mtaf.utils import project_root
from mtaf.utils.serialization import json_load
from mtaf.dataset.base_dataset import BaseDataset

DEFAULT_INFO_PATH = project_root / 'mtaf/dataset/idd_list/info7class.json'
INFO_PATH_7 = project_root / 'mtaf/dataset/idd_list/info7class.json'

class IDDDataSet(BaseDataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=DEFAULT_INFO_PATH, labels_size=None,
                 label_folder='',
                 num_classes=7):
        self.label_folder = label_folder
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)

        self.load_labels = load_labels
        if num_classes==7:
            self.info = json_load(INFO_PATH_7)
        else:
            self.info = json_load(info_path)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit' / self.set / name
        if self.label_folder is not '':
            label_file = self.label_folder + '/' + (name.split('/')[1])
        else:
            label_name = name.replace("leftImg8bit", "gtFine_labelids")
            label_file = self.root / 'gtFine' / self.set / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        label = self.get_labels(label_file)
        label = self.map_labels(label).copy()
        image = self.get_image(img_file)
        image = self.preprocess(image)
        return image.copy(), label, np.array(image.shape), name
