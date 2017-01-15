# coding: utf-8

from nltk import word_tokenize, sent_tokenize, FreqDist
import itertools
from utils import from_csv
import os.path
import optparse
import pickle
import time
import math
import numpy as np

class Model:
	UNKNOWN = 'UNKNOWN_TOKEN'
	START = 'START_SEQUENCE'
	END = 'END_SEQUENCE'
	def __init__(self, statements, batch_size=5000, vocabulary_size=2000, limit=0):
		if limit > 0:
			statements = statements[:limit]
		time_start = time.time()
		self.num_statements = len(statements)
		self.batch_ptr = 0
		print "[-] Tokenizing..."
		tokenized = [word_tokenize(sentence) for statement in statements for sentence in sent_tokenize(sanitize(statement))]
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
		self.padded = [list(itertools.chain(sequence, [0] * int(max(self.lengths) - len(sequence)))) for sequence in indexed]
		print "[-] Batches:",
		self.batch_size = min(len(self.padded), batch_size)
		self.num_batches = int(math.ceil(self.num_statements / batch_size))
		print "%s batches of size %s" % (self.num_batches, self.batch_size)
		time_diff = time.time() - time_start
		print "Parsed %s sentences in %s seconds" % (len(self.padded), time_diff)
		
	def translate(self, words):
		return ' '.join([self.lookup[x] for x in words])

	def batch(self, batch_ptr):
		start = self.batch_size * batch_ptr
		end = self.batch_size * (batch_ptr + 1)
		batch = self.padded[start:end]

		print "Generating batch from %s to %s, %s rows returned" % (start, end, len(batch))

		x = np.vstack([np.array(sequence[:-1]) for sequence in batch])
		y = np.vstack([np.array(sequence[1:]) for sequence in batch])

		print "Shape: [%s, %s]" % (x.shape, y.shape)

		return x, y

	def save(self, path):
		print "[-] Pickling..."
		pickle.dump(self, open(path, 'w'))

def restore(path):
	print "[-] Loading model..."
	assert os.path.exists(path), "Model file not found."
	time_start = time.time()
	model = pickle.load(open(path, 'rb'))
	time_diff = time.time() - time_start
	print "[-] Imported %s sentences in %s seconds" % (len(model.padded), time_diff)
	print '\a'
	return model

def create(path, limit=0, batch_size=500, vocabulary_size=2000):
	return Model(from_csv(path), limit=limit, batch_size=batch_size, vocabulary_size=vocabulary_size)

def sanitize(statement):
	result = unicode(statement[2], 'utf-8').lower().encode('ascii', 'ignore')
	# result = unicode(result.lower(), 'utf-8')
	return result


