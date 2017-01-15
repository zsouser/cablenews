csv:
	scrapy crawl msnbc -o "./csv/msnbc.csv" -t csv

model:
	python ./language/train.py  --csv "./csv/msnbc.csv" --checkpoint "./checkpoints/msnbc.rnn"

sample:
	python ./language/train.py  --csv "./csv/msnbc.csv" --checkpoint "./checkpoints/msnbc.rnn" --num_epochs 0

small:
	python ./language/train.py  --limit 5000 --batch_size 1000 --vocabulary_size 300 --csv "./csv/msnbc.csv" --output "./checkpoints/small.rnn"

medium:
	python ./language/model.py --limit 10000 --csv "./csv/msnbc.csv" --output "./models/medium.model"

large:
	python ./language/model.py --limit 100000 --csv "./csv/msnbc.csv" --output "./models/large.model"