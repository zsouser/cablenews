# -*- coding: utf-8 -*-
import scrapy

from scrapy.spiders import CrawlSpider, Rule, Request
from scrapy.linkextractors import LinkExtractor

from cablenews.items import StatementItem


class MsnbcSpider(CrawlSpider):
	name = "msnbc"
	allowed_domains = ["www.msnbc.com"]
	start_urls = (
		'http://www.msnbc.com/transcripts/',
	)
	rules = (
		Rule(LinkExtractor(allow=('transcripts/*', )), callback='parse_show'),
	)

	def parse_show(self, response):
		urls = response.xpath("//div[@class='item-list']//div[@class='item-list']//a//@href").extract()
		return [Request('http://www.msnbc.com' + url, callback=self.parse_month) for url in urls]

	def parse_month(self, response):
		urls = response.xpath("//div[@class='transcript-item']//a//@href").extract()
		return [Request('http://www.msnbc.com' + url, callback=self.parse_day) for url in urls]

	def parse_day(self, response):
		return self.parse_statements(response.xpath("//div[@itemprop='articleBody']//p/text()").extract())

	def parse_statements(self, statements):
		item = StatementItem(speaker="", statement="")
		for statement in statements:
			split = statement.split(':')
			first = split[0]

			if len(split) > 1:
				if first and first.upper() == first:
					yield item
					item = StatementItem()
					item["speaker"] = first.strip()
					item["statement"] = ' '.join(split[1:])
			else:
				if first and first[0] == '(':
					yield item
					item = StatementItem()
					item["speaker"] = ''
					item["statement"] = statement
				else:
					item["statement"] += ' ' + statement
		

