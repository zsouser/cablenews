import tensorflow as tf
import numpy as np
from tensorflow.python.ops import seq2seq

class RNN:
	def __init__(self, sequence_length, batch_size, vocabulary_size, embedding_size=128, num_units=64, learning_rate=0.01, bptt_truncate=4):
		print "Initialize RNN:"
		print "[-] Vocabulary size of %s" % (vocabulary_size)
		print "[-] Maximum sentence length of %s" % (sequence_length)
		print "[-] Embedding layer size %s" % (embedding_size)
		print "[-] LSTM Depth %s" % (num_units)
		print "[+] Initializing input..."

		self.X = tf.placeholder(tf.int32, [batch_size, sequence_length], name="X")
		self.Y = tf.placeholder(tf.int32, [batch_size, sequence_length], name="Y")
		
		
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			print "[+] Initializing Embedding Layer..."
			self.embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="W")
			inputs = tf.split(1, sequence_length, tf.nn.embedding_lookup(self.embedding, self.X))
			inputs = [tf.squeeze(item, [1]) for item in inputs]
			
		with tf.name_scope('softmax'):
			print "[+] Initializing Softmax layer..."
			softmax_W = tf.Variable(tf.random_uniform([num_units, vocabulary_size], -1.0, 1.0), name="softmax_W")
			softmax_b = tf.Variable(tf.constant(0.1, shape=[vocabulary_size]), name="softmax_b")
		
		with tf.name_scope('lstm'):
			print "[+] Initializing LSTM layer..."
			cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
	
			def loop(prev, next):
				prev = tf.matmul(prev, softmax_W) + softmax_b
				prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
				return tf.nn.embedding_lookup(self.embedding, prev_symbol)

		with tf.name_scope('output'):
			self.initial_state = cell.zero_state(batch_size, tf.float32)

			self.outputs, self.final_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop)
			# output = tf.reshape(tf.concat(1, self.outputs), [-1, num_units])
			output = tf.reshape(self.outputs, [-1, num_units])
			self.logits = tf.matmul(output, softmax_W) + softmax_b
			self.probs = tf.nn.softmax(self.logits)
			
		with tf.name_scope('loss'):
			print "[+] Initializing Loss layer..."
			
			loss = seq2seq.sequence_loss_by_example([self.logits],
				[tf.reshape(tf.concat(1, self.Y), [-1, sequence_length])],
				[tf.ones([batch_size * sequence_length])],
				vocabulary_size)

			self.cost = tf.reduce_sum(loss) / batch_size / sequence_length

		with tf.name_scope('optimization'):
			print "[+] Initializing Optimization layer..."
			self.learning_rate = tf.Variable(learning_rate, trainable=False)
			variables = tf.trainable_variables()
			gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, variables),
				bptt_truncate)
			optimizer = tf.train.AdamOptimizer(self.learning_rate)
			self.train = optimizer.apply_gradients(zip(gradients, variables))