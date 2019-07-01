import torch
import torchvision
import torch.utils.data as data
import os
train = '/home/user/snakes/train/'
validation = '/home/user/snakes/validation/'
proportion_train = 0.75
for class_ in os.listdir(train):
    class_path_train = os.path.join(train, class_)
    class_path_validation = os.path.join(validation, class_)
    samples_list = os.listdir(class_path_train)
    number_of_samples = len(os.listdir(class_path_train))
    random_indices = list(data.SubsetRandomSampler(range(0,number_of_samples)))
    validation_indices = random_indices[int(proportion_train*len(random_indices)):]
    for idx in validation_indices:
        os.rename(os.path.join(class_path_train, samples_list[idx]), os.path.join(class_path_validation, samples_list[idx]))