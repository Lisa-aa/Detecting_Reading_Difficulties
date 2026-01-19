# Introduction 

This folder contains all code for the study "Detecting Reading Difficulties Using Eye-Tracking Metrics and Machine Learning Models with the Purpose of Building Adaptive Tools for Low-Literacy Support". It also contains the results of training the models and some images to give more insight into the models and data.

# Content

- pre_processing.py : contains the code to create all featuresets of all the data and save them to files.
- load_data.py : contains functions to load the created featuresets to be used by the models.
- CNN.py : contains the two CNN models AlexNet and ResNet50 and functions for training and hyperparameter searching of these models. Also for ensemble model on top of CNNS.
- SVM.py : contains functions to train SVMs. Also for training an ensemble model on top of the SVMs.
- training_models.py : contains functions to hyperparameter seach train all variants of models needed for the study. Combines functions from CNN.py and SVM.py.
- testing_models.py : contains functions to load the trained models and evaluate them on the test data.

# On GitHub
The GitHub repository does not contain the participant data or models. The trained models are not included, since many models are trained, the folder with the trained models is too large.
