import numpy as np

from mtaf.dataset.base_dataset import BaseDataset


class SYNDataSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128),
                 num_classes=19):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)

        if num_classes==19:
            # map to cityscape's ids
            self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                                  15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                                  8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        elif num_classes==16:
             self.id_to_trainid = {3 : 0, 4 : 1, 2 : 2, 21 : 3, 5 : 4, 7 : 5,
                                   15 : 6, 9 : 7, 6 : 8, 1 : 9, 10 : 10, 17 : 11,
                                   8 : 12, 19 : 13, 12 : 14, 11 : 15}
        elif num_classes==7:
            # map to mapillary's ids
            self.id_to_trainid = {3: 0, # road - flat
                                  4: 0, # sidewalk - flat
                                  2: 1, # building - construction
                                  21: 1, # wall - construction
                                  5: 1, # fence - construction
                                  7: 2, # pole - object
                                  15: 2, # traffic light - object
                                  9: 2, # traffic sign - object
                                  6: 3, # vegetation - nature
                                  16: 0, # terrain - flat
                                  1: 4, # sky - sky
                                  10: 5, # person - human
                                  17: 5, # rider - human
                                  8: 6, # car - vehicle
                                  18: 6, # truck - vehicle
                                  19: 6, # bus - vehicle
                                  20: 6, # train - vehicle
                                  12: 6, # motorcycle - vehicle
                                  11: 6} # bicycle - vehicle
        else:
            raise NotImplementedError(f"Not yet supported {num_classes} for Synthia.")
    def get_metadata(self, name):
        img_file = self.root / 'RGB' / name
        label_file = self.root / 'GT' / 'CLASSES' / name
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
