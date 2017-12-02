from __future__ import print_function, division
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.serialization import load_lua
import os, random

PATH_TO_DATA_FOLDER = "C:\\Users\\Itamar Talmon\\Dropbox\\Masters\\SecondYear\\Deep Learning\\EX2\\EX2_data\\"
PATH_TO_AFLW = PATH_TO_DATA_FOLDER + "aflw\\"
PATH_TO_FDDB = PATH_TO_DATA_FOLDER + "fddb\\"
PATH_TO_FDDB_IMAGES = os.sep.join([PATH_TO_FDDB, 'images'])
PATH_TO_FDDB_IMAGE_PATHES_FILE = os.sep.join([PATH_TO_FDDB, 'FDDB-folds', 'FDDB-fold-01.txt'])
AFLW_12 = PATH_TO_AFLW + "aflw_12.t7"
AFLW_24 = PATH_TO_AFLW + "aflw_24.t7"
PATH_TO_PASCAL_FOLDER = "C:\\Users\\Itamar Talmon\\Downloads\\VOCtrainval_06-Nov-2007\\VOCdevkit\VOC2007\\"
PATH_TO_PASCAL_IMGS = PATH_TO_PASCAL_FOLDER + "JPEGImages\\"
PATH_TO_PASCAL_LABELS = PATH_TO_PASCAL_FOLDER + "ImageSets\\Main\\person_trainval.txt"

def load_t7_imgs(path):
    '''
    Loads the .t7 data files
    :param path: the path to the .t7 target file
    :return: torch Tensor with the img data
    '''
    o = load_lua(path)
    img_num = len(o.items())
    input_channels = len(o[1])
    img_w = len(o[1][0])
    img_h = len(o[1][0][0])
    inputs = torch.Tensor(img_num, input_channels, img_w, img_h)
    for k, v in o.items():
        # k - 1 is necessary since Lua is 1-indexed and Python is 0-indexed
        inputs[k - 1] = torch.Tensor(v)
    return inputs

def get_negative_samples(num_of_samples, crop_size):
    '''
    Sample non-person random crops from the PASCAL VOC dataset
    :param crop_size: the target patch size
    :return: torch Tensor with image patches
    '''
    inputs = torch.Tensor(num_of_samples, 3, crop_size, crop_size)

    img_file_list = os.listdir(PATH_TO_PASCAL_IMGS)
    person_labels = {}
    with open(PATH_TO_PASCAL_LABELS) as person_label_file:
        for line in person_label_file.read().strip().split('\n'):
            name, label = line.strip().split()
            person_labels[name + '.jpg'] = label == '1'
    # remove all person-labeled images
    non_person_images = [img for img in img_file_list if not person_labels[img]]
    i = 0
    while i < num_of_samples:
        image_file = random.choice(non_person_images)
        # read and normalize a random image
        f = io.imread(PATH_TO_PASCAL_IMGS + image_file) / 255
        fh, fw, _ = f.shape
        for _ in range(int(num_of_samples/len(non_person_images))):
            rx = random.randint(0, fw - crop_size)
            ry = random.randint(0, fh - crop_size)
            p = (f[ry: ry + crop_size, rx: rx + crop_size])
            # numpy reads H x W x Channels // torch reads Channels x H x W
            rp = np.rollaxis(p, 2)
            inputs[i, :, :, :] = torch.from_numpy(rp)
            i += 1
            if not (i < num_of_samples):
                break

    return inputs


def get_positive_train_and_test_sets(net_size, train_frac=0.8):
    """This functions reads the csv file and return randomly chosen test and train sets"""
    path = AFLW_12 if net_size == 12 else AFLW_24
    img_data = load_t7_imgs(path).numpy()
    # Split the data frame to train and test sets randomly
    np.random.shuffle(img_data)
    idx = int(train_frac*len(img_data))
    img_data = torch.from_numpy(img_data)
    train, test = img_data[:idx], img_data[idx:]
    return train, test


class AFLW(Dataset):
    """AFLW dataset."""

    def __init__(self, net_size, inputs_tensor, num_of_neg_samples=None):
        # if not specified, num_of_negative == num_of_positive
        num_of_neg_samples = len(inputs_tensor) if not num_of_neg_samples else num_of_neg_samples
        self.net_size = len(inputs_tensor[0, 0])
        assert (net_size in [12, 24]), "illegal net_size {0}!".format(net_size)
        self.positive_samples = inputs_tensor
        self.negative_samples = get_negative_samples(num_of_neg_samples, net_size)
        self.training_images = torch.cat([self.positive_samples, self.negative_samples], 0)
        self.training_labels = torch.cat([torch.ones(len(self.positive_samples)),
                                          torch.zeros(len(self.negative_samples))], 0).long()

    def __len__(self):
        return len(self.training_images)

    def __getitem__(self, idx):
        return self.training_images[idx], self.training_labels[idx]

    def show_image(self, idx):
        image, _ = self.__getitem__(idx)
        plt.imshow(image.reshape(-1, self.net_size, self.net_size))
        plt.pause(1)

def show_patch(p):
    i = np.rollaxis(p.numpy(), 2)
    i = np.rollaxis(i, 2)
    plt.imshow(i.astype(np.uint8))
    plt.pause(10)


def get_fddb_image_paths():
    with open(PATH_TO_FDDB_IMAGE_PATHES_FILE) as rf:
        path_list = rf.read().strip().split('\n')
    return path_list

