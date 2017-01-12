csv:
	scrapy crawl msnbc -o "./csv/msnbc.csv" -t csv

model:
	python ./language/model.py --csv "./csv/msnbc.csv" --output "./models/msnbc.model" --rnn

small:
	python ./language/model.py --limit 1000 --csv "./csv/msnbc.csv" --output "./models/small.model" --rnn

smallrnn:
	python ./language/model.py --pickle "./models/small.model" --rnn

rnn:
	python ./language/model.py --pickle "./models/msnbc.model" --rnn

