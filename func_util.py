import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import func_model 

def load_data(data_dir):
    
    #data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    # ImageFolder is a generic data loader where the images are arranged in this way:
    #root/dog/xxx.png
    #root/dog/xxy.png
    #root/dog/xxz.png

    #root/cat/123.png
    #root/cat/nsdf3.png
    #root/cat/asd932_.png
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader, train_data



# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # open image and get size for x & y axis
    pil_image = Image.open(image)
    pil_image_x = pil_image.size[0]
    pil_image_y = pil_image.size[1]
    
    #print(pil_image_x, pil_image_y)
    
    # scale image ==> shortest side 256 and retain aspect_ratio
    aspect = float(pil_image_x)/float(pil_image_y)
    
    if(pil_image_x > pil_image_y):
        size_y = 256
        size_x = 256 * aspect
    elif(pil_image_x < pil_image_y):
        size_x = 256
        size_y = 256 / aspect
    else:
        size_x = 256
        size_y = 256
        
    #print(size_x, size_y)
    
    # Resize PIL image
    pil_image.thumbnail((size_x, size_y))
        
    # crop out center 224x224 portion of image
    left = (size_x/2 - 112)
    top = (size_y/2 - 112)
    right = (size_x/2 + 112)
    bottom = (size_y/2 + 112)
    
    #print(left, top, right, bottom)
    

 
    # Crop out center 224x224 portion of image
    pil_image_cropped = pil_image.crop((left, top, right, bottom))
    
    # Converting PIL image to Numpy array
    np_image = np.array(pil_image_cropped)
    
    # Change color channels from int to float and dividng by 255 to make color channel values between 0 and 1
    np_image = np_image.astype('float')/255
    
    # Apply normalization
    np_image = (np_image - mean)/std
    
    # Flip dimensions (H, W, C) to (C, H, W)
    np_image = np_image.transpose((2,0,1))
    #print(np_image.shape)
    
    # Converting numpy.array to PyTorch tensor
    return torch.from_numpy(np_image)



