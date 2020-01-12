""" The purpose of this class is to resize and delete unusable images from the directory """
#!/usr/bin/env python
# importing required libraries
import glob
import os
import cv2
import numpy as np
from PIL import Image

# use enumerator in line 96

class PreProcessor:
    """ Class to resize and ommit images """

    def __init__(self, path_to_screenshot_folder, width, height, path_to_processed, logger):
        self.path_to_screenshot_folder = path_to_screenshot_folder
        self.width = width
        self.height = height
        self.path_to_processed = path_to_processed
        self.logger = logger

    def list_all_pictures(self, path_to_screenshot_folder):
        """ This function list all the images in directory """

        folder_to_string = str(path_to_screenshot_folder)

        # lock the image directory for further processing
        folder = glob.glob(
            folder_to_string
        )

        # for taking all pictures in the directory to a list
        image_path_list = []

        # for appending all screenshots to the defined list in the directory
        for folder_iterator in folder:
            for screenshot in glob.glob(folder_iterator + "/*.jpeg"):
                image_path_list.append(screenshot)

        return image_path_list

    def resize_pictures(self, path_to_screenshot_folder, path_to_processed, width, height):
        """ This function resizes all the images in directory with provided dimensions"""

        # call ListAllPictures() to initiate the screenshot list
        image_counter = 0
        list_image_paths = self.list_all_pictures(
            path_to_screenshot_folder
        )

        for image_counter in range(len(list_image_paths)):
            # variable for splitting image name from the path
            processed_image_name = str(
                list_image_paths[image_counter]
            )
            # for opening screenshots one by one in the image_path_list
            image = Image.open(
                list_image_paths[image_counter]
            )

            # splitting only to image name from the path, change depending on platform
            processed_image_name = processed_image_name.rsplit("\\", 1)[1]

            """
            optional to show current screenshot dimension
            current_width, current_height = image.size
            print("Image URL: " + str(list_image_paths[image_counter]) + "
            : Current Dimension - " + current_width + current_height )
            """

            image = image.resize(
                (width, height), Image.ANTIALIAS
            )  # resizing images to specified dimension using built-in library

            image.save(
                path_to_processed + str(processed_image_name)
            )  # saving screenshots with the same name and new dimension

    def clear_screenshots(self):
        """ This function clears the directory of original screenshots after processing """
        directory = glob.glob(self.path_to_screenshot_folder)
        for file in directory:
            os.remove(file)
        self.logger.warning("Removed Full-size Screenshots")

    def delete_white_pictures(self, path_to_processed):
        """ This function deletes all the iamges in directory that have too many white pixels """

        image_counter = 0
        count_deleted = 0

        # call ListAllPictures() to initiate the screenshot list
        list_image_paths = self.list_all_pictures(
            path_to_processed
        )

        for image_counter in range(len(list_image_paths)):
            # reading each screenshot as a grayscale image
            image = cv2.imread(
                list_image_paths[image_counter], cv2.IMREAD_GRAYSCALE
            )
            # defining white presence in an image
            number_white_pixels = np.sum(image == 255)

            # getting image dimensions for further calculation
            image_dimensions = image.shape
            height = image_dimensions[0]
            width = image_dimensions[1]
            dimensions = width * height

            # calculating white pixel presence in a screenshot
            white_presence = (number_white_pixels / dimensions) * 100

            # setting up threshold for white pixels in an image
            white_pixel_threshold = 90.00

            if white_presence > white_pixel_threshold:
                self.logger.debug(
                    "image number- " + str(image_counter) + ": " + "White percentage: ",
                    "{0:.2f}".format(white_presence) + "%",
                )
                if os.path.exists(list_image_paths[image_counter]):
                    self.logger.warning("Deleting File... ")
                    # for deleting images that have too much white pixels in it
                    os.remove(
                        list_image_paths[image_counter]
                    )
                    # for counting the number of deleted files
                    count_deleted += 1
            else:
                continue
