import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import rescale
import numpy as np
from nms import py_cpu_nms


class Det12(nn.Module):
    def __init__(self):
        super(Det12, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.fc = nn.Linear(in_features=16 * 4 * 4, out_features=16)
        self.fc1 = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        '''
        :summary:
            A simple convolutional net for 12 X 12 images
        :param x:
            BatchSize X InputChannels X 12 X 12 Tensor
        :return:
            The probabilities of the face detection or the last FC layer outputs (depends on the input arg)
        '''
        x = x.view(-1, 3, 12, 12)
        x = self.conv(x)
        x = F.relu(F.max_pool2d(x, kernel_size=3, stride=2))
        x = x.view(-1, 16 * 4 * 4)
        # this will be used in the Det24 Net
        x = F.relu(self.fc(x))
        x = self.fc1(x)
        x = F.softmax(x)
        return x


class FCN12(nn.Module):
    def __init__(self):
        super(FCN12, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1)

    def forward(self, x):
        '''
        :summary:
            A simple convolutional net for 12 X 12 images
        :param x:
            BatchSize X InputChannels X 12 X 12 Tensor
        :return:
            The probabilities of the face detection or the last FC layer outputs (depends on the input arg)
        '''
        #print(x.size())
        x = self.conv(x)
        #print(x.size())
        x = F.relu(F.max_pool2d(x, kernel_size=3, stride=2))
        #print(x.size())
        # this will be used in the Det24 Net
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        #print(x.size())
        x = F.softmax(x)
        #print(x.size())
        return x


class SimpleDetector():
    '''
    Simple detector with 12FCN net
    '''
    def __init__(self, net, scale_list=[0.5, 0.2, 0.1, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01], nms_threshold=0.8):
        self.net = net
        self.scale_list = scale_list
        self.nms_threshold = nms_threshold

    def detect(self, img):
        '''
        Run face detection on a single image
        :param img: the image as torch tensor 3 x H x W
        :return: list of bounding boxes of the detected faces
        '''
        # check if gray-scale
        if len(np.shape(img)) != 3:
            img = np.reshape(img, (np.shape(img)[0], np.shape(img)[1], 1))
            img = np.concatenate((img, img, img), axis=2)
        # print(img.shape)
        result_boxes = []
        for scale in self.scale_list:
            if (scale*min([img.shape[0], img.shape[1]])) < 20:
                break
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
                        xmin = int(w*(1/scale))
                        xmax = int((w + 12)*(1/scale))
                        ymin = int(h*(1/scale))
                        ymax = int((h + 12)*(1/scale))
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


class Det24(nn.Module):
    def __init__(self):
        super(Det24, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1)
        self.fc = nn.Linear(in_features=64 * 9 * 9, out_features=128)
        self.fc1 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        '''
        :summary:
            A simple convolutional net for 24 X 24 images
        :param x:
            BatchSize X InputChannels X 24 X 24 Tensor
        :return:
            The probabilities of the face detection or the last FC layer outputs (depends on the input arg)
        '''
        x = x.view(-1, 3, 24, 24)
        x = self.conv(x)
        x = F.relu(F.max_pool2d(x, kernel_size=3, stride=2))
        x = x.view(-1, 64 * 9 * 9)
        x = F.relu(self.fc(x))
        x = self.fc1(x)
        x = F.softmax(x)
        return x


class BetterDetector():
    '''
    Better detector with 24FCN net and 12-Detector
    '''
    def __init__(self, net, simple_detector, scale_list=[0.5, 0.2, 0.1, 0.07, 0.05, 0.04, 0.03, 0.02], nms_threshold=0.8):
        self.net = net
        self.simple_detector = simple_detector
        self.scale_list = scale_list
        self.nms_threshold = nms_threshold

    def detect(self, img):
        '''
        Run face detection on a single image
        :param img: the image as torch tensor 3 x H x W
        :return: list of bounding boxes of the detected faces
        '''
        bbox_from_12 = self.simple_detector.detect(img)
        img = np.rollaxis(img, 2).copy()
        result = []
        for box in bbox_from_12:
            window = img[:, int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2])]
            window = torch.autograd.Variable(torch.from_numpy(window).view(-1, 3, 24, 24)).float()
            output = self.net(window)
            # if the new detector agrees with the simple one, keep the bbox
            if output[:, 1, :, :] > 0.5:
                result.append([int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3]), output[:, 1, :, :]])

        # run global NMS
        if len(result):
            result = py_cpu_nms(np.array(result), self.nms_threshold)

        return result

