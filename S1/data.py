import torch
from numpy import load
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def mnist():
    folder = '/Users/maria/Documents/DTU/9. Semester/02476 Machine Learning Operations Jan 22/dtu_mlops/data/corruptmnist/'
    train0 = load(folder + 'train_0.npz')
    train1 = load(folder + 'train_1.npz')
    train2 = load(folder + 'train_2.npz')
    train3 = load(folder + 'train_3.npz')
    train4 = load(folder + 'train_4.npz')
    test = load(folder + 'test.npz')

    train_images = np.concatenate((train0['images'], train1['images'], train2['images'], train3['images'], train4['images']))
    train_labels = np.concatenate((train0['labels'], train1['labels'], train2['labels'], train3['labels'], train4['labels']))
    
    train_images_tensor = torch.Tensor(train_images)
    train_labels_tensor = torch.Tensor(train_labels).type(torch.LongTensor)

    train_data = TensorDataset(train_images_tensor,train_labels_tensor)
    train = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    test_images_tensor = torch.Tensor(test['images'])
    test_labels_tensor = torch.Tensor(test['labels']).type(torch.LongTensor)

    test_data = TensorDataset(test_images_tensor,test_labels_tensor)
    test = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    return train, test
