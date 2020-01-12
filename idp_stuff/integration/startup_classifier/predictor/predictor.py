""" Predict Probabilities """
#!/usr/bin/env python
# coding: utf-8
import os.path
import csv
import glob
from PIL import Image
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.transforms as transforms


class Net(nn.Module):
    """ Define the architecture for Neural Network"""
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 53 * 53, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, neural_net):
        """ define forward pass """
        # add sequence of convolutional and max pooling layers
        neural_net = self.pool(functional.relu(self.conv1(neural_net)))
        neural_net = self.pool(functional.relu(self.conv2(neural_net)))
        neural_net = self.dropout(neural_net)
        neural_net = neural_net.view(-1, 32 * 53 * 53)
        neural_net = functional.relu(self.fc1(neural_net))
        neural_net = self.dropout(functional.relu(self.fc2(neural_net)))
        neural_net = self.softmax(self.fc3(neural_net))
        return neural_net


class Predictor:
    """
    Predict the probability from the images
    """
    def __init__(self, path_of_processed_image, width, height, path_to_submission,
                 path_of_the_model, model_name, positive_threshold,
                 hashed_url_file_path, hashed_url_file_name):
        self.path_of_processed_image = path_of_processed_image
        self.width = width
        self.height = height
        self.path_to_submission = path_to_submission
        self.model_name = model_name
        self.path_of_the_model = path_of_the_model
        self.positive_threshold = positive_threshold
        self.hashed_url_file_path = hashed_url_file_path
        self.hashed_url_file_name = hashed_url_file_name

    def image_loader(loader, image_name):
        """ Load Images as tensor """
        image = Image.open(image_name)
        image = loader(image).float()
        image = torch.tensor(image, requires_grad=True)
        image = image.unsqueeze(0)
        return image

    def data_transform(width, height):
        """ Transform images """
        data_transforms = transforms.Compose(
            [
                transforms.Resize(width),
                transforms.CenterCrop(height),
                transforms.ToTensor(),
            ]
        )
        return data_transforms

    def clear_processed_screenshots(self):
        """ Delete all processed screenshots """
        directory = glob.glob(self.path_of_processed_image)
        for file in directory:
            os.remove(file)

    def predict(self, path_of_processed_image, path_of_submission,
    path_of_the_model, model_name, positive_threshold):
        if("alex" in model_name):
            # create a AlexNet CNN
            model = models.alexnet()
        elif("vgg" in model_name):
            # create a VGG16 CNN
            model = models.vgg16()
        else:
            # create a complete SCV2_CNN
            model = Net()
			
		# load the saved model	
        full_model_path = path_of_the_model + "/" + model_name

        model.load_state_dict(
            torch.load(full_model_path, map_location=lambda storage, loc: storage)
        )
        model.eval()

        csv_file = path_of_submission + "/" + model_name.rsplit(".", 1)[0] + ".csv"
        print(csv_file)

        # open csv to write
        with open(csv_file, "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",", lineterminator="\n")
            writer.writerow(["id", "probability", "prediction"])

        i = 0
        for filename in os.listdir(path_of_processed_image):
            pred_list = []
            if filename.endswith("jpeg"):
                # transfrom the image into data
                data_transforms = Predictor.data_transform(224, 224)
                path_of_single_image = (
                    path_of_processed_image + "\\" + filename
                )  # change this path according to OS
                single_image_data = Predictor.image_loader(
                    data_transforms, path_of_single_image
                )

                output = model(single_image_data)
                smax = nn.Softmax()
                smax_out = smax(output)[0]
                non_startup_prob = smax_out.data[0]
                startup_prob = smax_out.data[1]
                prob = startup_prob
                if non_startup_prob > startup_prob:
                    prob = 1 - non_startup_prob
                    prob = np.around(prob.cpu(), decimals=4)
                prob = np.around(prob.item(), decimals=4)

                if prob > positive_threshold:
                    prediction_label = "positive"
				elif prob > 0.45:
					prediction_label = "neutral"
                else:
                    prediction_label = "negative"

                if i % 10 == 0:
                    print(
                        str(i)
                        + "--"
                        + filename
                        + "--"
                        + str(prob)
                        + "--"
                        + prediction_label
                    )

                pred_list = [filename.replace(".jpeg", ""), prob, prediction_label]

                with open(csv_file, "a") as csvfile:
                    writer = csv.writer(csvfile, delimiter=",", lineterminator="\n")
                    writer.writerow(pred_list)

                i += 1

            full_hashed_url_file_path = (
                self.hashed_url_file_path + self.hashed_url_file_name
            )
            unlabeled = pd.read_csv(full_hashed_url_file_path)
            labels = pd.read_csv(csv_file)  # predicted csv

            merged = unlabeled.merge(labels, on="id")
            final_csv_file = (
                path_of_submission + "/" + model_name.rsplit(".", 1)[0] + "_final.csv"
            )
            merged.to_csv(final_csv_file, index=False)

        print("Done.")
