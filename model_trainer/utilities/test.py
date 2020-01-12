# file test.py
# author: Kazi Mahbub Mutakabbir
# Email:kazi.mahbub@tum.de
# date 03-11-2019

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from torch.autograd import Variable
from model_architecture.vanilla_cnn_2 import SimpleCNN
from utilities.loader import Loader
from GPUtil import showUtilization as gpu_usage
from time import time
from configuration import Configuration

def get_accuracy(outputs, labels):
        outputs = np.argmax(outputs, axis=1)
        #print(outputs)
        #print(labels)
        return np.sum(outputs==labels)/float(labels.size)

def testNet(net, test_dataset, device):
    print("Initial GPU Usage: ")
    gpu_usage()
    number_of_batches = len(test_dataset)
    test_start_time = time()
    print("Test started at: " + test_start_time)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_dataset:
            # inputs, labels = Variable(inputs), Variable(labels)
            inputs = Variable(inputs)
            inputs = inputs.to(device)
            # labels = labels.to(device)

            test_outputs = net(inputs)
            test_outputs = test_outputs.data.cpu().numpy()
            _, predicted = torch.max(test_outputs.data, 1)
            labels = labels.data.cpu().numpy()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # test_accuracy = get_accuracy(test_outputs, labels)
            print("Accuracy of network: %d" % (100 * correct / total))


config = Configuration()
loader = Loader(batch_size_train=config.batch_size_train,
                        batch_size_test = config.batch_size_test,
                        batch_size_validation = config.batch_size_validation)
training_dataset, validation_dataset, test_dataset = loader.get_datasets(config.training_data_path,
                                                                        config.test_data_path,
                                                                        config.validation_data_path)
device = torch.device("cuda" if torch.cuda.is_available() 
                        else "cpu")
model = SimpleCNN()
state_dict = torch.load(config.path_to_saved_model)
#print(state_dict)
model.load_state_dict(state_dict)
#print(model)
model.eval()
model.cuda()
testNet(model, test_dataset, device)
torch.cuda.empty_cache()