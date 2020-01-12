import os
from datetime import datetime

class Configuration():
    """ Initialize all defalut values """
    time = datetime.now().strftime('%Y-%m-%d %H_%M_%S')

    def __init__(self):
        # define all the file paths
        self.training_data_path = os.getcwd() + "\\data_for_training\\training_data"
        self.test_data_path = os.getcwd() + "\\data_for_training\\test_data\\"
        self.validation_data_path = os.getcwd() + "\\data_for_training\\validation_data"
        self.path_of_submission = os.getcwd() + "\\submissions\\"
        self.path_to_saved_model = os.getcwd() + "\\model_saved\\simpleCNN_" + self.time + ".pt"
        self.path_to_model_architecture = os.getcwd() + "\\model_architecture\\"
        self.path_to_tensorboard_log = os.getcwd() + "\\runs"

        # define all the file names
        self.url_file_name = "production_data.csv"
        self.hashed_url_file_name = "hashed_urls.csv"
        self.saved_model_name = "model_vanilla_cnn_1.pt"

        # set learning parameters
        self.batch_size_train = 32
        self.batch_size_test = 16
        self.batch_size_validation = 32
        self.learning_rate = 0.0001
        self.number_of_epochs = 10
        self.positive_threshold = 0.65
