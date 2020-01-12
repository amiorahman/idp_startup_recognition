import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Loader:
    def __init__(self, batch_size_train, 
                batch_size_test, batch_size_validation):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.batch_size_validation = batch_size_validation

    # Training Data Loader
    # Takes in a dataset and a training sampler for loading
    # num_workers deals with system memory and threads
    # this is the default data loader
    def get_datasets(self, training_data_path, test_data_path, validation_data_path):
        """
        Args:
            batch_size = number of images to be taken in a single batch
        """
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                                        )])
        # get training data
        train_data = datasets.ImageFolder(root=training_data_path, transform=transform)
        train_data_loader = data.DataLoader(train_data, batch_size=self.batch_size_train, shuffle=True,  num_workers=2)
        # get validation data
        validation_data = datasets.ImageFolder(root=validation_data_path, transform=transform)
        validation_data_loader = data.DataLoader(validation_data, batch_size=self.batch_size_validation, shuffle=True, num_workers=2)
        # get test data 
        test_data = datasets.ImageFolder(root=test_data_path, transform=transform)
        test_data_loader  = data.DataLoader(test_data, batch_size=self.batch_size_test, shuffle=True, num_workers=2)

        print("Total number of training data: ", len(train_data))
        print("Total number of validation data: ", len(validation_data))
        print("Total number of test data: ", len(test_data))
        print("Detected Classes are: ", train_data.class_to_idx)

        return train_data_loader, validation_data_loader, test_data_loader

    
    # Training Data Loader
    # Takes in a dataset and a training sampler for loading
    # num_workers deals with system memory and threads
    # this is an alternative data loading method and not used by default

    def get_train_loader(self, dataset, train_sampler):
        """
        Args:
            batch_size = number of images to be taken in a single batch
        """
        train_loader = data.DataLoader(dataset, batch_size=self.batch_size_train,
                                    sampler=train_sampler, num_workers=1,
                                    pin_memory = True)
        
        return train_loader

    # Validation Data Loader
    # Takes in a dataset and a validation sampler for loading
    # num_workers deals with system memory and threads
    # this is an alternative data loading method and not used by default

    def get_val_loader(self, dataset, val_sampler):
        val_loader = data.DataLoader(dataset, batch_size=self.batch_size_validation, 
                             sampler=val_sampler, num_workers=1,
                             pin_memory = True)
        return val_loader

    # Test Loader
    # Takes in a dataset and a test sampler for loading
    # num_workers deals with system memory and threads
    # this is an alternative data loading method and not used by default

    def get_test_loader(self, dataset, test_sampler):
        test_loader = data.DataLoader(dataset, batch_size=self.batch_size_test, 
                              sampler=test_sampler, num_workers=1,
                              pin_memory=True)
        return test_loader