import os.path
import optparse
import pickle
import tensorflow as tf
import numpy as np
import time

class RNN:
	def __init__(self, X, sequence_length, vocabulary_size, embedding_size=128, num_units=64):
		print "Initialize RNN:"
		print "[-] Vocabulary size of %s" % (vocabulary_size)
		print "[-] Maximum sentence length of %s" % (sequence_length)
		print "[-] Embedding layer size %s" % (embedding_size)
		print "[-] LSTM Depth %s" % (num_units)
		print "[+] Initializing input..."
		self.X = np.array(X)
		
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			print "[+] Initializing Embedding Layer..."
			W = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="W")
			self.embedded = tf.nn.embedding_lookup(W, X)
			self.embedded_expanded = tf.expand_dims(self.embedded, -1)
		
		with tf.name_scope('lstm'):
			print "[+] Initializing LSTM layer..."
			cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
			self.outputs, self.last_states = tf.nn.dynamic_rnn(
			    cell=cell,
			    inputs=self.embedded,
			    dtype=tf.float32)	

	def predict(self, n=1):
		prediction = tf.contrib.learn.run_n(
								{ "outputs" : self.outputs, "embedded" : self.embedded },
								n=n)
		return list(np.searchsorted(prediction[0]['embedded'][0][0], prediction[0]['outputs'][0][0]))

def rnn(model):
	with tf.Graph().as_default():
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		sess.run(tf.global_variables_initializer())
		with sess.as_default():
			tf.global_variables_initializer()

			rnn = RNN(X=model.padded, 
						sequence_length=model.max_length, 
						vocabulary_size=model.vocabulary_size)

			print '\a'
			try:
				while len(raw_input('Type anything to stop')) == 0:
					print model.translate(rnn.predict())
			except KeyboardInterrupt:
				print "Bye!"