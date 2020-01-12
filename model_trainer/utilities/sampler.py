import os
import math
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class Sampler():
    def __init__(self, train_ratio, validation_ratio, test_ratio):
        self.dataset = self.get_dataset()
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.n_training_samples, self.n_val_samples, self.n_test_samples = self.count_percentage(self.train_ratio,
                                                                                            self.validation_ratio,
                                                                                            self.test_ratio)

    # check for data imbalance
    # returns the total count of training, validation and test samples
    # count the number of training/test/validation samples automatically
    # this function requires all the positive and negative data to be placed in specific folders

    def count_percentage(self, train_ratio, validation_ratio, test_ratio):
        # count the number of total positive examples
        # file path is given inside os.walk params
        count_positive = sum([len(files) for (r, d, files) in
                            os.walk(os.getcwd() + '/test_input/pos/'
                            )])
        
        print('Total postive labeled images: ' + str(count_positive))
        
        count_negative = sum([len(files) for (r, d, files) in
                            os.walk(os.getcwd() + '/test_input/neg/'
                            )])
        print('Total negative labeled images: ' + str(count_negative))
        
        total_files = count_positive + count_negative
        
        if count_positive > count_negative:
            difference = count_positive - count_negative
            average = total_files / 2
            relative_percentage = difference / average * 100
            print('The Positive Dataset is relatively bigger by: ' \
                + '{0:.4f}'.format(relative_percentage) + '%')
        else:
            difference = count_negative - count_positive
            average = total_files / 2
            relative_percentage = difference / average * 100
            print('The Negative Dataset is relatively bigger by: ' \
                + '{0:.4f}'.format(relative_percentage) + '%')
        
        training_samples = math.floor(total_files * train_ratio)
        testing_samples = math.floor(total_files * test_ratio)
        validation_samples = math.floor(total_files * validation_ratio)
        
        return training_samples, validation_samples, testing_samples

    # populate the dateset and pass it to the train-test-validation loader
    # load the image dataset

    def get_dataset(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                                        )])
        dataset = datasets.ImageFolder("./test_input", transform = transform)
        return dataset

    # Create samplers to split the available training data into training, test and cross validation
    
    # Sampler for Training data
    def get_training_sampler(self):
        train_sampler = SubsetRandomSampler(np.arange(self.n_training_samples, dtype=np.int64))
        return train_sampler

    # Sampler for validation data
    def get_validation_sampler(self):
        val_sampler = SubsetRandomSampler(np.arange(self.n_training_samples, self.n_training_samples + self.n_val_samples, dtype = np.int64))
        return val_sampler
    
    # Sampler for test data
    def get_test_sampler(self):
        test_sampler = SubsetRandomSampler(np.arange(self.n_test_samples, dtype=np.int64))
        return test_sampler
    

