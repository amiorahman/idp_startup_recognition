""" Initializer Module """

import os

class Configuration():
    """ Initialize all defalut values """
    def __init__(self):
        # define all the file paths
        self.url_file_path = os.getcwd() + "\\urls\\"
        self.hashed_url_file_path = os.getcwd() + "\\urls\\hashed_urls\\"
        self.screenshots_path = os.getcwd() + "\\screenshots\\"
        self.path_of_submission = os.getcwd() + "\\submissions\\"
        self.path_of_the_model = os.getcwd() + "\\models\\"
        self.path_to_processed = os.getcwd() + "\\processedscreenshots\\"

        # define all the file names
        self.url_file_name = "classified_labels_new_yes.csv"
        self.hashed_url_file_name = "hashed_urls.csv"
        self.model_name = "model_scv2.pt"

        # set other parameters
        self.batch_size = 17
        self.width = 224
        self.height = 224
        self.positive_threshold = 0.5
