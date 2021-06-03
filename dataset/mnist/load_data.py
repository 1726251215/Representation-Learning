import numpy as np
from torch.utils.data import Dataset, DataLoader
from struct import unpack


def read_label(label_path):
    with open(label_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        label = np.fromfile(f, dtype=np.uint8)
    return label


def read_image(image_path):
    with open(image_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        # img = np.fromfile(f, dtype=np.uint8).reshape(num, 1, 28, 28).repeat(3, axis=1)
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img


def convert_one_hot(label):
    one_hot_label = np.zeros((label.shape[0], 10))
    for i, row in enumerate(one_hot_label):
        row[label[i]] = 1
    return one_hot_label


class MyDataSet(Dataset):
    def __init__(self, image_path, label_path):
        self.db = read_image(image_path)
        # self.label = convert_one_hot(read_label(label_path))
        self.label = read_label(label_path)

    def __getitem__(self, index):
        return {'image': self.db[index] / 255.0, 'label': self.label[index]}

    def __len__(self) -> int:
        return self.db.shape[0]
