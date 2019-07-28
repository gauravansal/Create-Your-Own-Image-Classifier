# Create Your Own Image Classifier Project

### Table of Contents

1. [Project Overview](#overview)
2. [Project Components](#components)
3. [Installation](#installation)
4. [File Descriptions](#files)
5. [Instructions](#instructions)
6. [Results](#results)
7. [Screenshots](#screenshots)
8. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview<a name="overview"></a>

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below.

![Flowers](https://github.com/gauravansal/Create-Your-Own-Image-Classifier/blob/master/assets/Flowers.png)

The project is broken down into multiple steps:
* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

## Project Components<a name="components"></a>

The Project is divided in the following two parts:

1. Deep Learning Model - 
In a Jupyter Notebook, a flower dataset of images is downloaded as a training, testing, and validation set. Then the datasets are transformed in order to increase accuracy as well as fit the input format for pre-trained networks. Resizing, cropping, random flipping are a few transformations. Next, densenet121 is chosen to use as the pre-trained network, and a new feed-forward classifier using the features of pre-trained network is defined. Both ReLU activation and dropout are used in the the classifier. After defining hyperparameters, such as number of epochs, the learning rate, etc. the model is trained on the training set. Training loss, validation loss, and accuracy are printed. After running the test set through the model, an accuracy of about 92% is achieved. A checkpoint saves the model, classifier, and its hyperparameters. A predict and check function are defined which output the top 5 possible flower species for a given image, along with their probabilities in a bar chart.

2. Command Line Application - 
The second portion of the project includes the 2 python files, “train.py” and “predict.py.” This application allows people to train a model on a dataset of images and then predict their classes from the command line. The train file uses the same NeuralNetwork class from Part 1, but now the user can choose either densenet121, vgg16, alexnet or resnet18 as the pre-trained network. Other parameters, such as number of epochs, number of hidden layers, etc. can be changed by the user. This file should output training loss, validation loss, and accuracy; as well as save a checkpoint. In the predict file, the checkpoint from the train file is loaded and then the top ‘k’ classes and their probabilities are printed.

## Installation<a name="installation"></a>

* Python 3.5+ (I used Python 3.6)
* Data Processing & Machine Learning Libraries: NumPy, Pandas, PyTorch
* Imaging Library: PIL(Pillow)
* Data Visualization: Matplotlib

## File Descriptions<a name="files"></a>

Below are the files/folders:
1. `data` - 
The data used specifically for this assignment is a flower database and is not provided in the repository as it's larger than what github allows. Nevertheless, feel free to create your own databases and train the model on them to use with your own projects. The structure of your data should be the following:
 - The data needs to be comprised of 3 folders namely, test, train and validate. Generally the proportions should be 70% training 10% validation and 20% test data.
- Inside the train, test and validation folders there should be folders bearing a specific number which corresponds to a specific category, clarified in the json file. For example if we have the image "a.jpg" and it is a rose it could be in a path like this /test/5/a.jpg and json file would be like this {...5:"rose",...}. Make sure to include a lot of photos of your catagories (more than 10) with different angles and different lighting conditions in order for the network to generalize better.
2. `cat_to_name.json` - In order for the network to print out the name of the flower a ".json" file is required. If you aren't familiar with json you can find information [here](https://www.json.org/). By using a ".json" file the data can be sorted into folders with numbers and those numbers will correspond to specific names specified in the ".json" file.
3. `Image Classifier Project.ipynb` - a Jupyter Notebook to perform the loading, training, testing and validation of image data.
4. `train.py` - Train a new network on a data set with train.py within Command Line Application.
5. `predict.py` -  Predict flower name from an image with predict.py along with the probability of that name within Command Line Application.
6. `checkpoint_model_pretrained_densenet.pth` - a PyTorch checkpoint to save the model and its hyperparameters.
7. `func_model.py` - python file containing all the functions used during model creation, training, validation, checkpoint creation and testing of image data.
8. `func_util.py` - python file containing all the functions for image data pre-processing.


## Instructions<a name="instructions"></a>
### ***Viewing the Jyputer Notebook***
In order to better view and work on the jupyter Notebook I encourage you to use nbviewer . You can simply copy and paste the link to this website and you will be able to edit it without any problem. Alternatively you can clone the repository using

```git clone https://github.com/gauravansal/Create-Your-Own-Image-Classifier/```
then in the command Line type, after you have downloaded jupyter notebook type

```jupyter notebook```
locate the notebook and run it.

### ***Command Line Application***
* Train a new network on a data set with ```train.py```
  * Basic Usage : ```python train.py data_directory```
  * Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  * Options:
    * Set directory to save checkpoints: ```python train.py data_dir --save_dir save_directory```
    * Choose arcitecture (alexnet, densenet121 or vgg16 available): ```pytnon train.py data_dir --arch "vgg16"```
    * Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20 ```
    * Use GPU for training: ```python train.py data_dir --gpu gpu```
    
* Predict flower name from an image with ```predict.py``` along with the probability of that name. That is you'll pass in a single image ```/path/to/image``` and return the flower name and class probability
  * Basic usage: ```python predict.py /path/to/image checkpoint```
  * Options:
    * Return top **K** most likely classes:``` python predict.py input checkpoint ---top_k 3```
    * Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    * Use GPU for inference: ```python predict.py input checkpoint --gpu```

The best way to get the command line input into the scripts is with the [argparse module](https://docs.python.org/3/library/argparse.html) in the standard library. You can also find [a nice tutorial for argparse here](https://pymotw.com/3/argparse/).

### ***Running the code on GPU***
As the network makes use of a sophisticated deep convolutional neural network the training process is impossible to be done by a common laptop. In order to train your models to your local machine you have three options:

1. **Cuda** -- If you have an NVIDIA GPU then you can install CUDA from [here](https://developer.nvidia.com/cuda-downloads). With Cuda you will be able to train your model however the process will still be time consuming
2. **Cloud Services** -- There are many paid cloud services that let you train your models like [AWS](https://aws.amazon.com/fr/) or  [Google Cloud](https://cloud.google.com/)
3. **Coogle Colab** -- [Google Colab](https://colab.research.google.com/) gives you free access to a tesla K80 GPU for 12 hours at a time. Once 12 hours have ellapsed you can just reload and continue! The only limitation is that you have to upload the data to Google Drive and if the dataset is massive you may run out of space.

However, once a model is trained then a normal CPU can be used for the predict.py file and you will have an answer within some seconds.

### ***Hyperparameters***
As you can see you have a wide selection of hyperparameters available and you can get even more by making small modifications to the code. Thus it may seem overly complicated to choose the right ones especially if the training needs at least 15 minutes to be completed. So here are some hints:
* By increasing the number of epochs the accuracy of the network on the training set gets better and better however be careful because if you pick a large number of epochs the network won't generalize well, that is to say it will have high accuracy on the training image and low accuracy on the test images. Eg: training for 12 epochs training accuracy: 85% Test accuracy: 82%. Training for 30 epochs training accuracy 95% test accuracy 50%.
* A big learning rate guarantees that the network will converge fast to a small error but it will constantly overshot
* A small learning rate guarantees that the network will reach greater accuracies but the learning process will take longer
* Densenet121 works best for images but the training process takes significantly longer than alexnet or vgg16
My settings were lr=0.001, dropout=0.3, epochs= 12 and my test accuracy was 92% with densenet121 as my feature extraction model.

### ***Pre-Trained Network***
The checkpoint.pth file contains the information of a network trained to recognise 102 different species of flowers. It has been trained with specific hyperparameters thus if you don't set them right the network will fail. In order to have a prediction for an image located in the path /path/to/image using pretrained model you can simply type ```python predict.py /path/to/image checkpoint.pth```

## Results<a name="results"></a>
 - 92% Accuracy of the network on the test images was produced.
 - [Entire Analysis can be found here: Developing an AI application(https://nbviewer.jupyter.org/github/gauravansal/Create-Your-Own-Image-Classifier/blob/master/Image%20Classifier%20Project.html)]


## Screenshots<a name="screenshots"></a>
***Inference for classification***
![Inference for classification](https://github.com/gauravansal/Create-Your-Own-Image-Classifier/blob/master/assets/inference_example.png)


## Licensing, Authors, and Acknowledgements<a name="licensing"></a>

<a name="license"></a>
### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
### Acknowledgements

This project was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025). 














