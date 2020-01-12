""" take screenshots and predict """
import asyncio
import time
# import all necessary class packages
from log.log_manager import configure_logger
from screenshot_taker.screenshot_taker import ScreenshotTaker
from preprocessor.preprocessor import PreProcessor
from predictor.predictor import Predictor
from configuration import Configuration
# only import nest_asyncio if running on a jupyter notebook
# import nest_asyncio


class StartupClassifier:
    """ Main pipeline class """
    def __init__(self):
        pass

    def pipeline(self):
        """  Main method """
        #start a timer that keeps track of total time needed
        start_time = time.time()
        # load all the necessary parameters from configuration file
        config = Configuration()
        #start the operation
        logger = configure_logger('default')
        logger.info("Operation started")
        # load the urls from the csv file and take full-size screenshots
        run_screenshot = ScreenshotTaker(
            config.url_file_path,
            config.url_file_name,
            config.hashed_url_file_path,
            config.hashed_url_file_name,
            logger
        )
        list_urls_hashed = run_screenshot.link_processor()

        # run the screenshotModule inside the event loop manager
        asyncio.get_event_loop().run_until_complete(
            run_screenshot.screenshot_module(
                list_urls_hashed[: config.batch_size], config.screenshots_path
            )
        )

        # resize and filter the screenshots
        preprocessor = PreProcessor(
            config.screenshots_path,
            config.path_to_processed,
            config.width,
            config.height,
            logger
        )
        preprocessor.resize_pictures(
            config.screenshots_path,
            config.path_to_processed,
            config.width,
            config.height
        )
        preprocessor.delete_white_pictures(config.path_to_processed)

        # delete the full size screenshots
        # preprocessor.clear_screenshots()

        # predict the resized images and label them accordingly
        predictor = Predictor(
            config.path_to_processed,
            config.path_of_submission,
            config.path_of_the_model,
            config.model_name,
            config.width,
            config.height,
            config.positive_threshold,
            config.hashed_url_file_path,
            config.hashed_url_file_name,
        )
        predictor.predict(
            config.path_to_processed,
            config.path_of_submission,
            config.path_of_the_model,
            config.model_name,
            config.positive_threshold,
        )

        # delete the processed screenshots
        # predictor.clear_processed_screenshots()

        end_time = time.time()
        logger.info("Successfully Completed")
        logger.info("Total time needed: " + str(end_time - start_time) + " seconds")


def main():
    """ Main pipeline execution block """
    startup_classifier = StartupClassifier()
    startup_classifier.pipeline()


if __name__ == "__main__":
    main()
