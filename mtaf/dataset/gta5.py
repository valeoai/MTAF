import numpy as np

from mtaf.dataset.base_dataset import BaseDataset


class GTA5DataSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128),
                 num_classes=19):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)
        self.crop_size = crop_size
        if num_classes==19:
            # map to cityscape's ids
            self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                                  19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                                  26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        elif num_classes==7:
            # map to mapillary's ids
            self.id_to_trainid = {7: 0, # road - flat
                                  8: 0, # sidewalk - flat
                                  11: 1, # building - construction
                                  12: 1, # wall - construction
                                  13: 1, # fence - construction
                                  17: 2, # pole - object
                                  19: 2, # traffic light - object
                                  20: 2, # traffic sign - object
                                  21: 3, # vegetation - nature
                                  22: 0, # terrain - flat
                                  23: 4, # sky - sky
                                  24: 5, # person - human
                                  25: 5, # rider - human
                                  26: 6, # car - vehicle
                                  27: 6, # truck - vehicle
                                  28: 6, # bus - vehicle
                                  31: 6, # train - vehicle
                                  32: 6, # motorcycle - vehicle
                                  33: 6} # bicycle - vehicle
        else:
            raise NotImplementedError(f"Not yet supported {num_classes} for GTA5.")
    def get_metadata(self, name):
        img_file = self.root / 'images' / name
        label_file = self.root / 'labels' / name
        return img_file, label_file

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        return image.copy(), label_copy.copy(), np.array(image.shape), name
