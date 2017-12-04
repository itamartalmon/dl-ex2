from skimage import io
from torchvision.transforms import Scale, ToPILImage, ToTensor
from input_handler import get_background_images, PATH_TO_PASCAL_IMGS
import torch
import numpy as np
import sys
from random import choice


def create_negative_examples(_12_detector, num_of_samples):

    result = torch.ByteTensor(num_of_samples, 3, 24, 24)

    to_pil = ToPILImage()
    scale_to_24 = Scale(size=24)
    to_tensor = ToTensor()

    background_images_list = get_background_images()
    i = 0
    while i < num_of_samples:
        image_path = choice(background_images_list)
        f = io.imread(PATH_TO_PASCAL_IMGS + image_path)
        # run the 12-Detector
        res = _12_detector.detect(f)
        f = torch.ByteTensor(torch.from_numpy(np.rollaxis(f, 2)))
        for box in res:
            cropped_image = f[:, int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2])]
            cropped_image = to_tensor(scale_to_24(to_pil(cropped_image)))
            result[i, :, :, :] = [result, cropped_image.view(-1, 3, 24, 24)]
            i += 1
            if not (i < num_of_samples):
                break
        sys.stdout.write('\rProcessed {}/{} samples'.format(i+1, num_of_samples))

    return result



