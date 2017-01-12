from nltk import word_tokenize, sent_tokenize, FreqDist
import itertools
from utils import from_csv
import os.path
import optparse
import pickle
import time
import rnn

class Model:
	UNKNOWN = 'UNKNOWN_TOKEN'
	START = 'START_SEQUENCE'
	END = 'END_SEQUENCE'
	def __init__(self, statements, vocabulary_size=5000, limit=0):
		if limit > 0:
			statements = statements[:limit]
		time_start = time.time()
		self.num_statements = len(statements)
		print "[-] Tokenizing..."
		tokenized = [word_tokenize(sentence) for statement in statements for sentence in sent_tokenize(unicode(statement[2], 'utf-8'))]
		self.lengths = [len(x) for x in tokenized]
		print "[-] Building Vocabulary..."
		self.vocabulary_size = vocabulary_size
		frequencies = FreqDist(itertools.chain(*tokenized))
		vocabulary = frequencies.most_common(vocabulary_size - 1)
		print "[-] Indexing tokens..."
		self.lookup = [x[0] for x in vocabulary] + [Model.UNKNOWN]
		index = dict((word, idx) for idx, word in enumerate(self.lookup))
		indexed = [[index[word] if word in index else index[Model.UNKNOWN] for word in stmt] for stmt in tokenized]
		print "[-] Padding sequences..."
		self.max_length = max(self.lengths)
		self.padded = [list(itertools.chain(sequence, [0] * int(self.max_length - len(sequence)))) for sequence in indexed]
		time_diff = time.time() - time_start
		print "Parsed %s sentences in %s seconds" % (len(self.padded), time_diff)
		print '\a'

	def translate(self, words):
		print ' '.join([self.lookup[x] for x in words])

def restore(path):
	print "[-] Loading model..."
	assert os.path.exists(path), "Model file not found."
	time_start = time.time()
	model = pickle.load(open(path, 'rb'))
	time_diff = time.time() - time_start
	print "[-] Imported %s sentences in %s seconds" % (len(model.padded), time_diff)
	print '\a'
	return model

def save(path):
	print "[-] Pickling..."
	pickle.dump(model, open(path, 'w'))

if __name__ == '__main__':
	parser = optparse.OptionParser()
	parser.add_option("-p", "--pickle", dest="pickle", help="Pickle file for existing model")
	parser.add_option("-c", "--csv", dest="filename", help="CSV Input for Language Model")
	parser.add_option("-o", "--output", dest="output", help="Path for serialized output")
	parser.add_option("-l", "--limit", dest="limit", default=0, help="Limit number of statements in the model", type="int")
	parser.add_option("-r", "--rnn", action="store_true", dest="rnn", default=False)

	options, args = parser.parse_args()
	model = None

	if options.pickle:
		assert os.path.exists(options.pickle), "No input file found"
		model = restore(options.pickle)
	else:
		assert os.path.exists(options.filename), "No input file found"
		data = from_csv(options.filename)
		model = Model(data, limit=options.limit)

	try:
		if options.rnn:
			rnn.rnn(model)
	finally:
		if options.output:
			save(options.output)