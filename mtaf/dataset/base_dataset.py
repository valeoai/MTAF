from pathlib import Path
import random
import numpy as np
from PIL import Image
from torch.utils import data


class BaseDataset(data.Dataset):
    def __init__(self, root, list_path, set_,
                 max_iters, image_size, labels_size, mean):
        self.root = Path(root)
        self.set = set_
        self.list_path = list_path.format(self.set)
        self.image_size = image_size
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean = mean
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file, label_file = self.get_metadata(name)
            self.files.append((img_file, label_file, name))

    def get_metadata(self, name):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))

    def preprocess_augmentation(self, image, label, crop_size, is_mirror=False):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(crop_size[1] - img_h, 0)
        pad_w = max(crop_size[0] - img_w, 0)
        #if pad_h > 0 or pad_w > 0:
        #    img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
        #        pad_w, cv2.BORDER_CONSTANT,
        #        value=(0.0, 0.0, 0.0))
        #    label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
        #        pad_w, cv2.BORDER_CONSTANT,
        #        value=(self.ignore_label,))
        #else:
        img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - crop_size[1])
        w_off = random.randint(0, img_w - crop_size[0])
        # roi = cv2.Rect(w_off, h_off, crop_size[1], crop_size[0]);
        image = np.asarray(img_pad[h_off : h_off+crop_size[1], w_off : w_off+crop_size[0]], np.float32)
        label = np.asarray(label_pad[h_off : h_off+crop_size[1], w_off : w_off+crop_size[0]], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if is_mirror:
            flip = np.random.choice(2)
            if flip:
                image = np.flip(image, axis=2).copy()
                label = np.flip(label, axis=1).copy()
        return image, label


    def get_image(self, file):
        return _load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_labels(self, file):
        return _load_img(file, self.labels_size, Image.NEAREST, rgb=False)


def _load_img(file, size, interpolation, rgb):
    img = Image.open(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    return np.asarray(img, np.float32)
