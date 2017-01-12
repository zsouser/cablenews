# -*- coding: utf-8 -*-
import scrapy


class FoxnewsSpider(scrapy.Spider):
    name = "foxnews"
    allowed_domains = ["www.foxnews.com"]
    start_urls = (
        'http://www.www.foxnews.com/',
    )

    def parse(self, response):
        pass
