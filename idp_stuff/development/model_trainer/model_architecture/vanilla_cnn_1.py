# create a SimpleCNN class that inherits pytorch.nn.module and implement a vanilla CNN

import torch.nn.functional as F
import torch.nn as nn

class SimpleCNN(nn.Module):

    # batch shape for input: (3, 32, 32), 3 = RGB channels

    def __init__(self):

        super(SimpleCNN, self).__init__()
        # super().__init__()

        # input channels = 3, output channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # input channels = 16, output channels = 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features
        self.fc1 = nn.Linear(32 * 55 * 55, 512)

        # 64 input features, 10 output features for our defrined classes
        self.fc2 = nn.Linear(512, 128)

        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        # compute the activation of the first convolution
        # size changes from (3, 32, 32) to (18, 32, 32)
        # print("Shape before 1st conv: " + str(x.shape))

        x = F.relu(self.conv1(x))

        # print("Shape after 1st conv: " + str(x.shape))
        # size changes from (18, 32, 32) to (18, 16, 16)

        x = self.pool(x)

        # print("Shape after 1st pool: " + str(x.shape))
        x = F.relu(self.conv2(x))

        # print("Shape after 2nd conv: " + str(x.shape))

        x = self.pool(x)
        # print("Shape after 2nd pool: " + str(x.shape))
        # Reshape data for the input layer of the net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension

        x = x.view(-1, 32 * 55 * 55)

        # print("Shape after view: " + str(x.shape))
        # computes the activation of the first fully connected layer
        # size changes from (1,4608) to (1,64)

        x = F.relu(self.fc1(x))

        # x = leaky_relu(self.fc1(x), negative_slope=0.01, inplace=False)
        # print("Shape after 1st fc1: " + str(x.shape))
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1. 64) to (1, 10)

        x = F.relu(self.fc2(x))
        # print("Shape after fc2: " + str(x.shape))
        x = self.fc3(x)
        return (x)
