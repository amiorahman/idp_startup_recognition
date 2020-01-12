import numpy as np
import pandas as pd
import os
import csv
import glob
import math

def count_percentage(train_ratio, valid_ratio, test_ratio):
	count_positive = sum([len(files) for r, d, files in os.walk("/home/amio/Downloads/idp_screenshots/yes/resized/yes/")])
	print("Total postive labeled images: " + str(count_positive))

	count_negative = sum([len(files) for r, d, files in os.walk("/home/amio/Downloads/idp_screenshots/no/resized/no/")])
	print("Total negative labeled images: " + str(count_negative))

	total_files = count_positive + count_negative

	if(count_positive > count_negative):
		difference = count_positive - count_negative
		average = total_files / 2

		relative_percentage = (difference / average) * 100

		print("The Positive Dataset is relatively bigger by: " + "{0:.2f}".format(relative_percentage) + "%")

	else:
		difference = count_negative - count_positive
		average = total_files / 2

		relative_percentage = (difference / average) * 100

		print("The Negative Dataset is relatively bigger by: " + "{0:.2f}".format(relative_percentage) + "%")

	training_samples = math.floor(total_files * train_ratio)
	testing_samples = math.floor(total_files * test_ratio)
	validation_samples = math.floor(total_files * valid_ratio)

	print("Total Training Samples: " + str(training_samples) + "\n" 
		+ "Total Testing Samples: " + str(testing_samples) + "\n"
		+ "Total Validation Samples: " + str(validation_samples))

	return training_samples, testing_samples, validation_samples

if __name__ == '__main__':
	count_percentage(0.5, 0.3, 0.2)