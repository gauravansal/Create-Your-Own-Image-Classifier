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
import func_util

# Build and train a new feed-forward classifier using the features of pre-trained network

def network_setup(arch, hidden_layer_size, dropout, learning_rate):
    # Load a pre-trained model and determine input_size for the network classifier
    if arch=='vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch=='densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif arch=='alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    else:
        print("'{}' is not a valid model. Model options are - vgg16,densenet121,or alexnet".format(arch))
        raise ValueError('Unexpected network architecture', arch)
        

    # Output size of the classifier would be 102 as there are 102 categories of the flower    
    output_size = 102
    
    # Freezing parameters so we don't backprop through them   
    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
            ('input', nn.Linear(input_size, hidden_layer_size)),
            ('relu1', nn.ReLU()),
            ('drop_p1',nn.Dropout(dropout)),
            ('hidden_layer', nn.Linear(hidden_layer_size, output_size)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))
            
    model.classifier = classifier
        
    # Defining Loss function and Optimisation method for the model classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)        


    return model, optimizer, criterion


# Implement a function for the validation pass
def validation(model, validloader, criterion, gpu):
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:

        # Move input and label tensors to the GPU if it is available
        if torch.cuda.is_available() and gpu:
            images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy


        

# defining Training function, so they can be used later. Training only model classifier with a pre-trained network    
def do_deep_learning_training(model, optimizer, criterion, epochs, print_every, trainloader, gpu, validloader):
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = print_every

    # Move model to gpu if it is available
    if torch.cuda.is_available() and gpu:
        model.to('cuda')

    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1
        
            # Move input and label tensors to the GPU if it is available
            if torch.cuda.is_available() and gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
        
        
            optimizer.zero_grad()
        
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, gpu)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
                running_loss = 0
            
                # Make sure training is back on
                model.train()
    print("-------------- Finished training -----------------------")
    
    
    
# Defining accuracy function to check validation on the test set
def check_accuracy_on_test(model, testloader, gpu):    
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # Move input and label tensors to the GPU if it is available
            if torch.cuda.is_available() and gpu:            
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
    
def save_checkpoint(model, optimizer, save_dir, arch, hidden_layer_size, dropout, learning_rate, epochs, train_data):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing
    This function saves the model at a specified by the user path
    '''
    if arch=='vgg16':
        input_size = 25088
    elif arch=='densenet121':
        input_size = 1024
    elif arch=='alexnet':
        input_size = 9216
   
    output_size = 102
    
    model.class_to_idx = train_data.class_to_idx
    #model.to('cpu')
    
    checkpoint = {'arch': arch,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layer_size': hidden_layer_size,
                  'dropout': dropout,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer_state': optimizer.state_dict()}
    
    torch.save(checkpoint, save_dir)
    print("-------------checkpoint saved-------------------")
    print("Path of checkpoint saved is: {}".format(save_dir))
    print("Use path '{}' while predicting the image".format(save_dir))
    

    
# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath, gpu):
    if torch.cuda.is_available() and gpu:
        checkpoint = torch.load(filepath)
        model, _, _ = network_setup(checkpoint['arch'],
                              checkpoint['hidden_layer_size'],
                              checkpoint['dropout'],
                              checkpoint['learning_rate'])
        model.load_state_dict(checkpoint['state_dict'])
        
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        model, _, _ = network_setup(checkpoint['arch'],
                              checkpoint['hidden_layer_size'],
                              checkpoint['dropout'],
                              checkpoint['learning_rate'])
        model.load_state_dict(checkpoint['state_dict'])
        
    return model, checkpoint['class_to_idx']




# Implement the code to predict the class from an image file
def predict(image_path, model, top_k, idx_to_class):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Make sure network is in eval mode for inference
    model.eval()
    
    # cpu mode
    # model.cpu()
    
    '''
    if torch.cuda.is_available():
        if gpu:
            model.to('cuda')
        elif not gpu:
            model.to('cpu')
    else:
        model.to('cpu')
    '''  

    
    # load image as torch.Tensor
    image = func_util.process_image(image_path)
    
    # Unsqueeze returns a new tensor with a dimension of size one inserted at the specified position
    # https://pytorch.org/docs/0.3.0/torch.html#torch.unsqueeze
    image = image.unsqueeze(0)
    #print(image.shape)
    #image = image.float().to('cuda')
    image = image.float()
    
    # Turning off gradients
    # (not needed with evaluation mode?)
    with torch.no_grad():
        output = model.forward(image)
        result = torch.topk(output, top_k)
        
        probability = F.softmax(result[0].data, dim=1).numpy()[0]
        classes_idx = result[1].data.numpy()[0]
        
        probs = []
        for i in probability:
            probs.append(round(i, 8))
    
        classes = [idx_to_class[x] for x in classes_idx]
      
    return probs, classes, classes_idx