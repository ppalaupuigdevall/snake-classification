#!/usr/bin/env python
# coding: utf-8
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn as nn
import tensorboardX
import numpy as np
from sklearn.metrics import f1_score
import os
import copy
import fastai
import matplotlib.pyplot as plt
import pylab
import pickle

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
print("Device: " + str(torch.cuda.get_device_name(0)))


class ImageFolderId(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderId, self).__getitem__(index), self.imgs[index] #return image path

train_dir = '/home/user/snakes/train/'
validation_dir = '/home/user/snakes/validation/'

# Define training parameters 
size_batch = 16

# Like Albert Pumarola said in class, normalize data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

# Define datasets
# Now the definition of the dataset for the trianing samples is a little bit different than before, we need to know
# the class path to build a confusion matrix
training_set = ImageFolderId(train_dir, transforms.Compose([transforms.RandomSizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize]))
validation_set = torchvision.datasets.ImageFolder(validation_dir, transforms.Compose([transforms.RandomSizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize]))
                                    
# Define dataloaders                               
train_loader = torch.utils.data.DataLoader(training_set, batch_size=size_batch, shuffle=True, num_workers=2, pin_memory=True)
val_loader = torch.utils.data.DataLoader(validation_set, batch_size=size_batch, shuffle=True, num_workers=2, pin_memory=True)

C = len(training_set.classes)
number_of_training_samples = len(training_set)
print("Number of classes C = " + str(C))
class_balance = torch.empty(C)
i=0
for cl in training_set.classes:
    # We want to penalize more the classes that are less frequent
    class_balance[i] = 1/len(os.listdir(os.path.join(train_dir, cl)))/number_of_training_samples
    i += 1

normalization_factor = class_balance.sum()
class_balance /= normalization_factor
class_balance = class_balance.to(device)

print("Weights provided to the loss:")
print(class_balance)

criterion = nn.CrossEntropyLoss(weight=class_balance)

model = torchvision.models.resnet50(pretrained=True, progress=True)

num_ftrs = model.fc.in_features

# Add dropout layers
model.layer4[1].relu = nn.Sequential(
    nn.Dropout2d(0.15),
    nn.ReLU(inplace=True)
)
model.layer4[2].relu = nn.Sequential(
    nn.Dropout2d(0.15),
    nn.ReLU(inplace=True)
)
# Change the last layer to classify 45 classes instead of 1000 and add dropout
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, C)
)
print(model)

# Now load the pretrained model with dropout
model_weights_path = '/home/user/finetuning/resnet50_snakes_drop_Ep_28_Acc_0.626_F_0.509.pth'
model_weights = torch.load(model_weights_path)
model.load_state_dict(model_weights)
# Now set the model to evaluation mode
model.eval()

# Now make a forward pass through all the training set and build a confusion matrix saving each sample-id
model = model.to(device)

"""Now let us compute the confusion matrix
The confusion matrix will contain all image paths that are missclassified, the reason is that later on we will implement a Siamese network. And we must know the image names of the hard examples that our network missclassified. Therefore, we will save two matrices: 
1. Matrix CxC where the rows are the indices for the actual classes and the columns the indices of the predicted classes. In this matrix we will store the names of the images in order to pick them later in pairs. Named confusion_matrix
2. Matrix CxC containing the number of examples that were correectly/wrong classified. A typical confusion matrix. Named cm

To do this, we need to define a new dataset that gives us the image names when loading them so we can save them.
"""
# # The confusion matrix will contain all ids that are missclassified
# confusion_matrix = [[[] for col in range(45)] for row in range(45)]
# for inputs, labels in train_loader:
#     idxs = labels[0]
#     labels = labels[1]
#     inputs = inputs[0].to(device)
#     outputs = model(inputs)
#     _, preds = torch.max(outputs, 1)
#     preds = preds.cpu()
#     indices = torch.eq(labels, preds).numpy()
#     #print("Labels:")
#     #print(labels)
#     #print(preds)
#     wrong_classification = np.where(indices == 0)
#     #print(wrong_classification[0])
#     for i in wrong_classification[0]:
#         #print("Sample number " + str(i)+ " has been missclassified")
#         #print("It was a sample of class: " + str(labels[i].item()))
#         #print("But it has been classified as: " + str(preds[i].item()))
#         #print("The name of the image is: " + str(idxs[i]))
#         confusion_matrix[labels[i]][preds[i].item()].append(idxs[i])
    
# with open('/home/user/confusion_matrix', 'wb') as f:
#     pickle.dump(confusion_matrix, f)
# #print(confusion_matrix)
# print("Finished confusion matrix computation")

# Once the confusion matrix is created, a new dataset to sample pairs has to be created
class PairsDataset(torch.utils.data.Dataset):
    
    def __init__(self, confusion_mat_paths, confusion_mat_counter):
        self.confusion_matrix_paths = confusion_mat_paths
        self.confusion_matrix_counter = confusion_mat_counter

    # We override __getitem_method
    def __getitem__(self,index):
        # We'll sample by columns or rows randomly. 0: row, 1: col
        
        if(row_or_col == 0):
            # We sample along an entire row (improve intra class predictions)
            length_of_non_null_row = np.where(self.confusion_matrix_counter[rnd_idx, :] != 0)
            rnd_columns = np.random.randint(0,length_of_non_null_row, 2)
            number_of_images = len(self.confusion_matrix_paths[rnd_idx][rnd_column])
            random_image = np.random(0, number_of_images, 1)
            






