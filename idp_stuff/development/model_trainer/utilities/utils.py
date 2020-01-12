import torch
import torchvision
import torch.backends.cudnn
import torch.optim as optim
import torch.nn as nn
import time
import sys
from torch.autograd import Variable
from GPUtil import showUtilization as gpu_usage
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utilities.loader import Loader
from numpy.random import rand

class Utilities():

    def __init__(self, device):
        #super(Utilities, self).__init__()
        self.device = device

    # optimize CUDA tasks using cuDNN

    def cudnn(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # define the loss and opitmizer functions
    # we use the CrossEntropyLoss and ADAM as Optimizer

    def createLossAndOptimizer(self,net, learning_rate):
        # Loss function
        loss = nn.CrossEntropyLoss()
        #out_act = nn.Sigmoid()
        #loss = nn.BCELoss()

        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        return (loss, optimizer)

    # calculate the accuracy of predictions
    def get_accuracy(self, out, labels):
        outputs = np.argmax(out, axis=1)
        #print(outputs)
        #print(labels)
        return np.sum(outputs==labels)/float(labels.size)

    # Train the CNN and calculate training/validation loss

    def trainNet(self, device, net, number_of_epochs, learning_rate,
                 training_dataset, validation_dataset, path_to_tensorboard_log, path_to_saved_model):

        print("Initial GPU Usage")
        gpu_usage()

        # Get Training Data
        number_of_batches = len(training_dataset)

        # Create our loss and optimizer functions
        loss, optimizer = self.createLossAndOptimizer(net, learning_rate)

        # Keep track of time
        training_start_time = time.time()

        print("GPU Usage before starting the first epoch")
        gpu_usage()
        print(number_of_epochs)

        # initialize the tensorboard
        # in the command line, navigate to the root folder of the project and then type:
        # tensorboard --logdir=runs
        # after launching it, navigate to the following website in the browser:
        # http://localhost:6006/
        writer = SummaryWriter()
        # get some random training images
        dataiter = iter(training_dataset)
        images, labels = dataiter.next()

        # create grid of images
        img_grid = torchvision.utils.make_grid(images)

        # show images
        # matplotlib_imshow(img_grid, one_channel=True)

        # write to tensorboard
        writer.add_image('idp_sr_training_images', img_grid)
        writer.add_graph(net, images)

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in net.state_dict():
            print(param_tensor, "\t", net.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])
        
        # transfer the network into the GPU
        net.to(device)
        
        # assign platform's maximum 
        max_val = sys.maxsize
        # assign the max value as lowest validation loss
        lowest_validation_loss  = max_val
        # Loop for number_of_epochs
        number_of_minibatches = 1
        
        for epoch in range(number_of_epochs):
            print("inside for loop")
            train_loss = 0.0
            total_val_loss = 0
            # accuracy = 0
            print_every = number_of_batches // 10
            start_time = time.time()
            total_train_loss = 0.0
            print("GPU Usage in epoch: ", epoch)
            gpu_usage()
            net.train()
            for i, data in enumerate(training_dataset, 0):
                # Get inputs
                inputs, labels = data
                #print(labels)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Wraps them in a Variable object
                inputs, labels = Variable(inputs), Variable(labels)

                # Set the parameter gradients to zero
                # And make the forward pass, calculate gradient, do backprop
                optimizer.zero_grad()
                outputs = net(inputs)
                #outputs = out_act(outputs)
                #labels = labels.view(-1,1)
                #labels = labels.float()
                loss_size = loss(outputs, labels)
                train_loss += loss_size.data
                total_train_loss += loss_size.data
                outputs = outputs.data.cpu().numpy()
                labels= labels.data.cpu().numpy()
                current_accuracy = self.get_accuracy(outputs, labels)
                #print(current_accuracy)
                del (inputs)
                del (labels)
                loss_size.backward()
                optimizer.step()
                # gpu_usage()
                
                if (i + 1) % (print_every + 1) == 0:
                    current_avg_loss = train_loss/print_every
                    print("Epoch {}, {:d}% \t Train loss: {:.4f} took: {:.4f}s".format(
                        epoch + 1, int(100 * (i + 1) / number_of_batches),
                        current_avg_loss,
                        time.time() - start_time))
                    writer.add_scalar('Mini-batch Training Loss', current_avg_loss, number_of_minibatches)
                    writer.add_scalar('Mini-batch accuracy:', current_accuracy, number_of_minibatches)
                    number_of_minibatches+=1

                    print("GPU Usage after 10th batch:")
                    gpu_usage()
                    # Reset running loss and time
                    train_loss = 0.0
                    start_time = time.time()
            
            # At the end of the epoch, do a pass on the validation set

            writer.add_histogram('conv1.bias', net.conv1.bias, epoch+1)
            writer.add_histogram('conv1.weight', net.conv1.weight, epoch+1)
            writer.add_histogram('conv1.weight.grad', net.conv1.weight.grad, epoch+1)
            net.eval()

            with torch.no_grad():
                for inputs, labels in validation_dataset:
                    # Wrap tensors in variables
                    inputs, labels = Variable(inputs), Variable(labels)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # Forward pass
                    val_outputs = net(inputs)
                    # val_outputs = out_act(val_outputs)
                    # labels = labels.view(-1, 1)
                    # labels = labels.float()
                    val_loss_size = loss(val_outputs, labels)
                    total_val_loss += val_loss_size.data
                    val_outputs = val_outputs.data.cpu().numpy()
                    labels = labels.data.cpu().numpy()
                    val_accuracy = self.get_accuracy(val_outputs, labels)
                    writer.add_scalar('Validation accuracy: ', val_accuracy, epoch + 1)

            current_validation_loss = total_val_loss / len(validation_dataset)
            print("validation loss = {:.4f}".format(current_validation_loss))
            if current_validation_loss < lowest_validation_loss:
                lowest_validation_loss = current_validation_loss
                torch.save(net.state_dict(), path_to_saved_model)
                print("Saving model..., path: ", path_to_saved_model)
            else:
                print("Validation loss increased. Keeping the previous model.")

        writer.close()
        print("Training finished. Took: {:.4f}s".format(time.time() - training_start_time))
    

    # calculate output tensor size for a given input image, kernel size, stride and padding

    def outputSize(self, in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
        return(output)

    # create backend for tensorboard

    def initialize_tensorboard(self, path_to_tensorboard_log, trainloader, net):
        writer = SummaryWriter(path_to_tensorboard_log)
        
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        # create grid of images
        img_grid = torchvision.utils.make_grid(images)

        # show images
        # matplotlib_imshow(img_grid, one_channel=True)

        # write to tensorboard
        writer.add_image('idp_sr_images', img_grid)
        writer.add_graph(net, images)
        return writer