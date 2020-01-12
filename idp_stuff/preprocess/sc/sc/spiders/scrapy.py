import json
import base64
import scrapy
from scrapy_splash import SplashRequest
from scrapy.spiders import Spider
import time
import random
import pandas as pd
import string

class ExtractSpider(scrapy.Spider):

    #df = pd.read_csv('/home/amio/idp_sr/nostart.csv')
    
    name = 'extract'
    imgCounter = 0 
     
    def start_requests(self):
        df = pd.read_csv('/home/amio/idp_sr/yesstart.csv')
        init_list = df["URL"].tolist()
        scheme = 'http://'
        url_list = [scheme + x for x in init_list]
        start_urls = url_list
        #start_urls = ['http://stackoverflow.com','http://youtube.com','http://innospot.de']
        #print(start_urls)
        print(type(start_urls))
        splash_args = {
            'html': 1,
            'png': 1,
            'width': 1080,
            'render_all': False,
        }

        for imgCounter in range(len(start_urls) + 1):
            yield SplashRequest(start_urls[imgCounter], self.parse_result, endpoint='render.json', args=splash_args)
            time.sleep(2)
            imgCounter += 1
            print(imgCounter)

    #def parse(self,reposnse):
        #pass

    def parse_result(self, response):
        imgdata = base64.b64decode(response.data['png'])
        filename = '/home/amio/idp_sr/yesStartUp/' + str(random.randint(1000000,2000000)) + '.png'

        with open(filename, 'wb') as f:
            f.write(imgdata)

    #def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    #    return ''.join(random.choice(chars) for _ in range(size))

    #def parse_result(self, response):
    #    imgdata = base64.b64decode(response.data['png'])
    #    char_set = string.ascii_uppercase + string.digits
    #    filename = ''.join(random.sample(char_set*6, 6))+ '.png'

    #    with open(filename, 'wb') as f:
    #        f.write(imgdata)
        