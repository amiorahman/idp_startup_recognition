""" Take screenshots """
#!/usr/bin/env python
# coding: utf-8

# Manage imports
import hashlib
import csv
import pandas as pd
from pyppeteer import launch
import urllib3


class ScreenshotTaker:
    """ Take screenshots from the URLs in the given csv file """
    def __init__(self, url_file_path, url_file_name,
                 hashed_url_file_path, hashed_url_file_name, logger):
        self.url_file_path = url_file_path
        self.url_file_name = url_file_name
        self.hashed_url_file_path = hashed_url_file_path
        self.hashed_url_file_name = hashed_url_file_name
        self.list_urls = []
        self.logger = logger

    def link_processor(self):
        """Formats the input from a csv into a flattened list and returns it"""
        df_urls = pd.read_csv(
            self.url_file_path + self.url_file_name, sep=" ", header=None
        )
        self.list_urls = df_urls.values.flatten().tolist()
        self.list_urls = ["http://" + s for s in self.list_urls]
        self.logger.debug("Total size: " + str(len(self.list_urls)))
        self.logger.debug(self.list_urls[:30])
        list_urls_hashed = []
        # save hashed website address to a csv
        for url in self.list_urls:
            list_urls_hashed.append(hashlib.md5(url.encode()).hexdigest())

        # key_value = dict(zip(list_urls_hashed, self.list_urls))
        headers = ["id", "url"]
        with open(self.hashed_url_file_path
                  + self.hashed_url_file_name, "w", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(headers)
            writer.writerows(zip(list_urls_hashed, self.list_urls))

        return list_urls_hashed

    async def screenshot_module(self, list_urls_hashed, screenshots_path):
        """
        This is the main screenshot taking function.
        Here, we first divide the input into multiple batches
        to help with error recovery and performance.
        Then, we launch the chromedriver (it automatically downloads
        the Chromedriver in case there is none in the system).
        The parameters and timeout settings can be customized if needed.
        We close the chromedriver instance at the end of each run to and relaunch for the next run.
        """

        number_of_urls = len(list_urls_hashed)
        # Find out the total number of loops needed for current input
        loops, extra_runs = divmod(number_of_urls, 1000)
        self.logger.debug("total loops: ")
        self.logger.debug(loops)
        self.logger.debug("Extra runs: ")
        self.logger.debug(extra_runs)

        # suppress certificate verification warning
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # get the chromedriver extension in case it's not installed already
        for j in range(loops + 1):
            self.logger.debug("Launching a new Chromium instance")
            browser = await launch(
                {
                    "args": ["--no-sandbox", "--disable-setuid-sandbox"],
                    "ignoreHTTPSErrors": "true",
                    "dumpio": "false",
                    "userDataDir": ".data",  # define a temporary cache storage for faster loads
                }
            )
            start = 0
            end = 1
            if (loops - j + 1) == 1:
                start = j * 1000
                end = j * 1000 + extra_runs
            else:
                # start at the n*1000th site, go upto 999
                start = j * 1000
                end = j * 1000 + 1000

            # go to the sites and take screenshots
            for i in range(start, end):
                # open a new page/window
                page = await browser.newPage()
                await page.setViewport(
                    {"width": 1280, "height": 720, "deviceScaleFactor": 1}
                )
                self.logger.debug(str(i) + ". Now visiting: " + self.list_urls[i])

                # set idle timeout
                try:
                    await page.goto(
                        self.list_urls[i],
                        {
                            "networkIdle2Timeout": 5000,
                            "waitUntil": "load"
                        },
                    )
                    self.logger.debug("Reached Page.")
                    """
                    taking a screenshot of the whole page,
                    possible to experiment with x/y cooridnates
                    """
                    await page.screenshot(
                        {
                            "path": screenshots_path + list_urls_hashed[i] + ".jpeg",
                            "type": "jpeg",
                        }
                    )

                    self.logger.debug("Screenshot Taken and Saved.")

                    # close the page
                    await page.close()
                    self.logger.debug("Page Closed.")

                except:
                    # Handle exception
                    self.logger.critical("There's an issue with this page: " + self.list_urls[i])
                    """
                    reduce CPU usage by closing the page,
                    otherwise it isn't picked up by the garbage collector
                    """
                    await page.close()
                    self.logger.critical(
                        "Closing the unresponsive page and moving to the next one."
                    )
                    continue

            # close the browser
            self.logger.debug("Closing the browser and setting it for relaunch.")
            await browser.close()
