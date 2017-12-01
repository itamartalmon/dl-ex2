import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.misc import imresize
from skimage.transform import rescale
import numpy as np
from nms import py_cpu_nms


class Det12(nn.Module):
    def __init__(self):
        super(Det12, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.fc = nn.Linear(in_features=16 * 4 * 4, out_features=2)

    def forward(self, x):
        '''
        :summary:
            A simple convolutional net for 12 X 12 images
        :param x:
            BatchSize X InputChannels X 12 X 12 Tensor
        :param get_last_fc:
            Boolean that states if we wish to return the last FC output instead of Sigmoid (False as defaults)
        :return:
            The probabilities of the face detection or the last FC layer outputs (depends on the input arg)
        '''
        x = x.view(-1, 3, 12, 12)
        x = self.conv(x)
        x = F.relu(F.max_pool2d(x, kernel_size=3, stride=2))
        x = x.view(-1, 16 * 4 * 4)
        # this will be used in the Det24 Net
        from_det12 = self.fc(x)
        x = F.sigmoid(from_det12)
        return x


class FCN12(nn.Module):
    def __init__(self):
        super(FCN12, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=4, stride=1)

    def forward(self, x):
        '''
        :summary:
            A simple convolutional net for 12 X 12 images
        :param x:
            BatchSize X InputChannels X 12 X 12 Tensor
        :param get_last_fc:
            Boolean that states if we wish to return the last FC output instead of Sigmoid (False as defaults)
        :return:
            The probabilities of the face detection or the last FC layer outputs (depends on the input arg)
        '''
        #print(x.size())
        x = self.conv(x)
        #print(x.size())
        x = F.relu(F.max_pool2d(x, kernel_size=3, stride=2))
        #print(x.size())
        # this will be used in the Det24 Net
        x = self.conv2(x)
        #print(x.size())
        x = F.sigmoid(x)
        #print(x.size())
        return x


class SimpleDetector():
    '''
    Simple detector with 12FCN net
    '''
    def __init__(self, net, scale_list=[0.5, 0.2, 0.1, 0.07, 0.05], nms_threshold=0.5):
        self.net = net
        self.scale_list = scale_list
        self.nms_threshold = nms_threshold

    def detect(self, img):
        '''
        Run face detection on a single image
        :param img: the image as torch tensor 3 x H x W
        :return: list of bounding boxes of the detected faces
        '''
        #check if gray
        if len(np.shape(img)) != 3:
            img = np.reshape(img, (np.shape(img)[0], np.shape(img)[1], 1))
            img = np.concatenate((img, img, img), axis=2)
        # print(img.shape)
        result_boxes = []
        for scale in self.scale_list:
            resized_image = rescale(img, scale, mode='constant', preserve_range=True)
            resized_image = np.rollaxis(resized_image, 2).copy()
            resized_image = np.uint8(resized_image / 255)
            resized_image = torch.autograd.Variable(torch.from_numpy(resized_image).view(-1, *resized_image.shape)).float()
            output = self.net(resized_image)
            # output size is 1 X 2 X H X W
            heatmap = output[:, 1, :, :] # take the probability of detecting Face class ( 1 X H X W )
            heatmap = heatmap.view(heatmap.size()[-2], heatmap.size()[-1]) # resize to matrix form ( H X W )
            preds = heatmap > 0.5 # 1 is we predict a face, 0 o/w
            H, W = preds.size()
            bboxes = []
            for h in range(H):
                for w in range(W):
                    if preds.data[h, w] == 1:
                        score = heatmap.data[h, w]
                        # print(score)
                        xmin = int(w*(1/scale))
                        xmax = int((w + 12)*(1/scale))
                        ymin = int(h*(1/scale))
                        ymax = int((h + 12)*(1/scale))
                        # print(ymin, ymax, xmin, xmax)
                        croped_img = img[ymin: ymax, xmin: xmax]
                        # print(croped_img.shape)
                        bboxes.append([xmin, ymin, xmax, ymax, score])
            # run NMS per scale
            if len(bboxes):
                bboxes = py_cpu_nms(np.array(bboxes), self.nms_threshold)
            # print(len(bboxes))
            result_boxes += bboxes
        # print(result_boxes)
        return result_boxes





