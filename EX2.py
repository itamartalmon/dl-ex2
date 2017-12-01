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

from input_handler import AFLW, get_train_and_test_sets, get_fddb_image_paths, PATH_TO_FDDB_IMAGES
from models import Det12, FCN12, SimpleDetector
from skimage import io


def train_model(model, criterion, optimizer, dset_loaders, dset_sizes, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0
    losses = {'train' : [], 'val': []}

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
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
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.data.long().view(preds.size(0)) == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

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
    num_of_epochs = 20
    batch_size = 64
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    pre_load = time.time()
    print('Getting AFLW data...')
    dataset, testDataset = get_train_and_test_sets(net_size, 0.8)
    print('Creating the Data Loaders...')
    trainloader = DataLoader(AFLW(net_size, dataset), batch_size, shuffle=True)
    testloader = DataLoader(AFLW(net_size, testDataset), batch_size, shuffle=True)
    dset_loaders = {'train': trainloader, 'val': testloader}
    dset_sizes = {'train': len(trainloader.dataset), 'val': len(testloader.dataset)}
    print('Loading time : {}'.format(time.time() - pre_load))
    best_model, losses = train_model(model, criterion, optimizer, dset_loaders, dset_sizes, num_epochs=num_of_epochs)
    torch.save(best_model, "Det12.t7")


def Q2():
    '''
    train FCN-12-Net, Test detection
    '''
    if os.path.isfile("FCN12.t7"):
        best_model = torch.load("FCN12.t7")
    else:
        model = FCN12()
        net_size = 12
        num_of_epochs = 20
        batch_size = 64
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        pre_load = time.time()
        print('Getting AFLW data...')
        dataset, testDataset = get_train_and_test_sets(net_size, 0.8)
        print('Creating the Data Loaders...')
        trainloader = DataLoader(AFLW(net_size, dataset), batch_size, shuffle=True)
        testloader = DataLoader(AFLW(net_size, testDataset), batch_size, shuffle=True)
        dset_loaders = {'train': trainloader, 'val': testloader}
        dset_sizes = {'train': len(trainloader.dataset), 'val': len(testloader.dataset)}
        print('Loading time : {}'.format(time.time() - pre_load))
        best_model, losses = train_model(model, criterion, optimizer, dset_loaders, dset_sizes, num_epochs=num_of_epochs)
        torch.save(best_model, "FCN12.t7")

    # run detector and output results
    d = SimpleDetector(best_model)
    img_list = get_fddb_image_paths(PATH_TO_FDDB_IMAGES)
    n = len(img_list)
    with open("results.txt", 'w') as f:
        for idx, img in enumerate(img_list):
            sys.stdout.write("\rProcesing image number {0}/{1} : {2}".format(idx, n, img[len(PATH_TO_FDDB_IMAGES):]))
            sys.stdout.flush()
            res = d.detect(io.imread(img))
            f.write(img[len(PATH_TO_FDDB_IMAGES):] + '\n')
            f.write(str(len(res)) + '\n')
            for r in res:
                s = ' '.join([str(x) for x in r])
                f.write(s + '\n')




# Q1()

Q2()

