import time
import copy
import os
import sys

# import torchfile
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from plot_utils import plot_learning_curves
from input_handler import AFLW, get_positive_train_and_test_sets, get_negative_samples, get_fddb_image_paths, PATH_TO_FDDB_IMAGES
from models import Det12, FCN12, SimpleDetector, Det24, BetterDetector
from negative_mining import create_negative_examples

from skimage import io
from PIL import Image, ImageDraw

def train_model(model, criterion, optimizer, dset_loaders, dset_sizes, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0
    losses = {'train' : [], 'val': []}

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        '''if not (epoch % 50):
            lr = optimizer.param_groups[0]['lr'] * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print('Updated LR to be {}'.format(lr))'''

        # Each epoch has a training and validation phase
        for phase in ['val', 'train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)
                loss = criterion(outputs.view(-1, 2).float(), labels.long())

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.data.long().view(preds.size(0)) == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            losses[phase].append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model, losses

def Q1():
    '''
    train and test 12-Net, print and plot the results, save the net
    '''
    model = Det12()
    net_size = 12
    batch_size = 128
    num_of_epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    pre_load = time.time()
    print('Getting AFLW data...')
    dataset, testDataset = get_positive_train_and_test_sets(net_size, train_frac=0.8)
    print('Creating the Data Loaders...')
    trainloader = DataLoader(AFLW(net_size, dataset, get_negative_samples(net_size, len(dataset))), batch_size, shuffle=True)
    testloader = DataLoader(AFLW(net_size, testDataset, get_negative_samples(net_size, len(testDataset))), batch_size, shuffle=True)
    dset_loaders = {'train': trainloader, 'val': testloader}
    dset_sizes = {'train': len(trainloader.dataset), 'val': len(testloader.dataset)}
    print('Loading time : {}'.format(time.time() - pre_load))
    best_model, losses = train_model(model, criterion, optimizer, dset_loaders, dset_sizes, num_of_epochs)
    torch.save(best_model, "Det12.t7")
    plot_learning_curves(losses['train'], losses['val'], 'Detection12Net')


def Q2(reload=True, output_detected_images=False):
    '''
    train FCN-12-Net, save results, test detection and output to file
    '''
    if os.path.isfile("FCN12.t7") and reload:
        best_model = torch.load("FCN12.t7")
    else:
        model = FCN12()
        net_size = 12
        num_of_epochs = 500
        batch_size = 128
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        pre_load = time.time()

        print('Getting AFLW data...')
        dataset, testDataset = get_positive_train_and_test_sets(net_size, 0.8)

        print('Creating the Data Loaders...')
        trainloader = DataLoader(AFLW(net_size, dataset, get_negative_samples(len(dataset), net_size)), batch_size, shuffle=True)
        testloader = DataLoader(AFLW(net_size, testDataset, get_negative_samples(len(dataset), net_size)), batch_size, shuffle=True)
        dset_loaders = {'train': trainloader, 'val': testloader}
        dset_sizes = {'train': len(trainloader.dataset), 'val': len(testloader.dataset)}

        print('Loading time : {}'.format(time.time() - pre_load))
        best_model, losses = train_model(model, criterion, optimizer, dset_loaders, dset_sizes, num_epochs=num_of_epochs)
        torch.save(best_model, "FCN12.t7")
        plot_learning_curves(losses['train'], losses['val'], 'FCN12Net')

    # run detector and output results
    d = SimpleDetector(best_model, nms_threshold=0.8)
    img_list = get_fddb_image_paths()
    n = len(img_list)

    with open("fold-01-out.txt", 'w', newline='\n') as f:
        for idx, img in enumerate(img_list):

            sys.stdout.write("\rProcesing image number {0}/{1} : {2}".format(idx+1, n, img))
            sys.stdout.flush()

            if os.name == 'nt':  # change path separator if we run from windows
                path = os.sep.join([PATH_TO_FDDB_IMAGES, img.replace('/', '\\') + '.jpg'])
            else:
                path = os.sep.join([PATH_TO_FDDB_IMAGES, img + '.jpg'])

            # run the 12-Detector
            res = d.detect(io.imread(path))

            if output_detected_images:
                if not os.path.exists('outputs'):
                    os.mkdir('outputs')
                i = Image.open(path)
                for box in res:
                    ImageDraw.Draw(i).rectangle((box[0], box[1], box[0] + box[2], box[1] + box[3]), outline="red")
                i.save(os.sep.join(["outputs", path.split(os.sep)[-1]]))
                i.close()

            f.write(img + '\n')
            f.write(str(len(res)) + '\n')
            for r in res:
                s = ' '.join([str(x) for x in r])
                f.write(s + '\n')


def Q3():
    '''
    train and test 24-Net, print and plot the results, save the net
    '''
    model = Det24()
    net_size = 24
    batch_size = 128
    num_of_epochs = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    pre_load = time.time()

    print('Getting AFLW data...')
    dataset, testDataset = get_positive_train_and_test_sets(net_size, train_frac=0.8)

    print('Mining Negative Samples...')
    if os.path.exists('mined_negative_samples_for_24.t7'):
        negative_sampled = torch.load('mined_negative_samples_for_24.t7')
    else:
        m = torch.load('FCN12.t7')
        d = SimpleDetector(m, scale_list=[0.2, 0.1, 0.05], nms_threshold=0.7)
        negative_sampled = create_negative_examples(d, num_of_samples=len(dataset)+len(testDataset))
        torch.save(negative_sampled, 'mined_negative_samples_for_24.t7')
        print(negative_sampled.size())

    print('Creating the Data Loaders...')
    trainloader = DataLoader(AFLW(net_size, dataset, negative_sampled[:len(dataset)]),
                             batch_size, shuffle=True)
    testloader = DataLoader(AFLW(net_size, testDataset, negative_sampled[len(dataset):]),
                            batch_size, shuffle=True)
    dset_loaders = {'train': trainloader, 'val': testloader}
    dset_sizes = {'train': len(trainloader.dataset), 'val': len(testloader.dataset)}
    print('Loading time : {}'.format(time.time() - pre_load))

    best_model, losses = train_model(model, criterion, optimizer, dset_loaders, dset_sizes, num_of_epochs)
    torch.save(best_model, "Det24.t7")
    plot_learning_curves(losses['train'], losses['val'], 'Detection24Net')

def Q4(reload=True, output_detected_images=False):
    '''
    test Better Detector and output to file
    '''
    if os.path.isfile("FCN12.t7") and reload:
        m12 = torch.load("FCN12.t7")
    else:
        # train 12
        pass
    simple_d = SimpleDetector(m12, nms_threshold=0.8)

    if os.path.isfile("Det24.t7") and reload:
        m24 = torch.load("Det24.t7")
    else:
        # train 24
        pass

    d = BetterDetector(m24, simple_d, nms_threshold=0.8)

    img_list = get_fddb_image_paths()
    n = len(img_list)

    with open("fold-01-out-24.txt", 'w', newline='\n') as f:
        for idx, img in enumerate(img_list):

            sys.stdout.write("\rProcesing image number {0}/{1} : {2}".format(idx+1, n, img))
            sys.stdout.flush()

            if os.name == 'nt':  # change path separator if we run from windows
                path = os.sep.join([PATH_TO_FDDB_IMAGES, img.replace('/', '\\') + '.jpg'])
            else:
                path = os.sep.join([PATH_TO_FDDB_IMAGES, img + '.jpg'])

            # run the Better-Detector
            res = d.detect(io.imread(path))

            if output_detected_images:
                if not os.path.exists('outputs'):
                    os.mkdir('outputs')
                i = Image.open(path)
                for box in res:
                    ImageDraw.Draw(i).rectangle((box[0], box[1], box[0] + box[2], box[1] + box[3]), outline="red")
                i.save(os.sep.join(["outputs", path.split(os.sep)[-1]]))
                i.close()

            f.write(img + '\n')
            f.write(str(len(res)) + '\n')
            for r in res:
                s = ' '.join([str(x) for x in r])
                f.write(s + '\n')


# Q1()

# Q2(reload=True, output_detected_images=True)

# Q3()

Q4()


