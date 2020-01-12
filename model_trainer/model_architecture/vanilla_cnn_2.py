# create a SimpleCNN class that inherits pytorch.nn.module and implement a vanilla CNN

# file vanilla_cnn_2.py
# author: Kazi Mahbub Mutakabbir
# Email:kazi.mahbub@tum.de
# date 17-10-2019

import torch.nn.functional as F
import torch.nn as nn

class SimpleCNN(nn.Module):

    # batch shape for input: (3, 32, 32), 3 = RGB channels

    def __init__(self):

        super(SimpleCNN, self).__init__()
        # super().__init__()

        ### first convolutional layer ###
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        # batch normalization
        self.batch_norm_1 = nn.BatchNorm2d(16)
        # reLU activation
        self.relu = nn.ReLU()
        # pooling layerZ
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # kernel size after 1st conv layer: 111*111*16

        ### 2nd convolutional layer ###
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        # batch normalization layer
        self.batch_norm_2 = nn.BatchNorm2d(32)
        # pooling layer
        # 1 padding added to get whole value of kernel dimension
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # kernel size after 2nd conv layer: 55*55*32

        ### 3rd convolutional layer ###
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        # batch normalization layer
        self.batch_norm_3 = nn.BatchNorm2d(64)
        # pooling layer
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # kernel size after 3rd conv layer: 26*26*64

        ### 4th convolutional layer ###
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        # batch normalization layer
        self.batch_norm_4 = nn.BatchNorm2d(128)
        # pooling layer
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # kernel size after 4th convolutional layer: 12*12*128

        ### 5th convolutional layer ###
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        # batch normalization layer
        self.batch_norm_5 = nn.BatchNorm2d(256)
        # pooling layer
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # kernel size after 5th convolutional layer: 5*5*256

        # fully connected layer 1
        self.fc1 = nn.Linear(in_features=256 * 5 * 5, out_features=4096)
        self.batch_norm_6 = nn.BatchNorm1d(4096)
        # add a dropout to reduce overfitting
        #self.dropout = nn.Dropout(p=0.3)

        # fully connected layer 2
        self.fc2 = nn.Linear(in_features=4096, out_features=1024)
        # add a dropout to reduce overfitting
        self.dropout = nn.Dropout(p=0.3)

        # fully connected layer 3
        self.fc3 = nn.Linear(1024, 256)
        # add a dropout
        self.dropout = nn.Dropout(p=0.5)

        # output layer
        self.fc4 = nn.Linear(256,2)


    def forward(self, x):
        # compute the activation of the first convolutional layer
        out = self.conv1(x)
        out = self.batch_norm_1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        
        # compute the activation of the second convolutional layer
        out = self.conv2(out)
        out = self.batch_norm_2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        
        # compute the activation of the third convolutional layer
        out = self.conv3(out)
        out = self.batch_norm_3(out)
        out = self.relu(out)
        out = self.maxpool3(out)

        # compute the activation of the fourth convolutional layer
        out = self.conv4(out)
        out = self.batch_norm_4(out)
        out = self.relu(out)
        out = self.maxpool4(out)

        # compute the activation of the fourth convolutional layer
        out = self.conv5(out)
        out = self.batch_norm_5(out)
        out = self.relu(out)
        out = self.maxpool5(out)
        # print("Shape after conv: " + str(out.shape))
        # Reshape data for the input layer of the net
        # Recall that the -1 infers this dimension

        out = out.view(-1, 256 * 5 * 5)

        # last conv layer -> 1st fc layer
        out = self.fc1(out)
        out = self.batch_norm_6(out)
        out = self.relu(out)
        #out = self.dropout(out)

        # 1st fc layer -> 2nd fc layer
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # 2nd fc layer -> 3rd fc layer
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)

        # 3rd fc layer -> output layer
        out = self.fc4(out)
        #out = self.relu(out)
        #out = self.dropout(out)

        return (out)
    
