csv:
	scrapy crawl msnbc -o "./csv/msnbc.csv" -t csv

model:
	python ./language/model.py --csv "./csv/msnbc.csv" --output "./models/msnbc.model" --rnn

small:
	python ./language/model.py --limit 1000 --csv "./csv/msnbc.csv" --output "./models/small.model" -b 200
	python ./language/train.py --model ./models/small.model --output "./checkpoints/small.rnn" --checkpoint "./checkpoints/small.rnn"
	python ./language/sample.py --model ./models/small.model --checkpoint "./checkpoints/small.rnn"

medium:
	python ./language/model.py --limit 10000 --csv "./csv/msnbc.csv" --output "./models/medium.model"

large:
	python ./language/model.py --limit 100000 --csv "./csv/msnbc.csv" --output "./models/large.model"