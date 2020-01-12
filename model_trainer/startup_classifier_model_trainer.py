# import the necessary libraries
import time
import torch
from model_architecture.vanilla_cnn_2 import SimpleCNN
from configuration import Configuration
from utilities.utils import Utilities
from utilities.loader import Loader

class ModelTrainer():
    def __init__(self):
        pass

    def pipeline(self):
        print("pipeline class")
        #start a timer that keeps track of total time needed
        start_time = time.time()
        # load all the necessary parameters from configuration file
        config = Configuration()
        # define the default device
        # cuda:0 if a compatible CUDA GPU has been found, CPU otherwise
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        utils = Utilities(device)
        utils.cudnn()
        loader = Loader(batch_size_train=config.batch_size_train,
                        batch_size_test = config.batch_size_test,
                        batch_size_validation = config.batch_size_validation)
        training_dataset, validation_dataset, test_dataset = loader.get_datasets(config.training_data_path,
                                                                                config.test_data_path,
                                                                                config.validation_data_path)
        
        net = SimpleCNN()
        print(net)
        # start training the model
        utils.trainNet(device, net, config.number_of_epochs, config.learning_rate,
                        training_dataset, validation_dataset, config.path_to_tensorboard_log, config.path_to_saved_model)

        end_time = time.time()
        print("Total time elapsed: " + str(end_time - start_time) + " seconds")
        # clear GPU Memory buffer after all tasks have been completed
        torch.cuda.empty_cache()

# create the main function and call the pipeline class
def main():
    """ Main pipeline execution block """
    model_trainer = ModelTrainer()
    model_trainer.pipeline()

if __name__ == "__main__":
    main()