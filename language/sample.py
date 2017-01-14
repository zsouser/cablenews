from model import restore, Model
import optparse
import os.path
import tensorflow as tf
import numpy as np
from rnn import RNN

parser = optparse.OptionParser()
parser.add_option("-m", "--model", dest="model", help="Pickle file for existing model")
parser.add_option("-c", "--checkpoint", dest="checkpoint", help="Checkpoint file for RNN")

options, args = parser.parse_args()
model = None

assert os.path.exists(options.model), "Model not found"
print "[+] Importing language model"
language_model = restore(options.model)
sequence_length = language_model.max_length - 1

with tf.Graph().as_default():
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	with sess.as_default():
		rnn = RNN(sequence_length=sequence_length, 
			batch_size=language_model.batch_size,
			vocabulary_size=language_model.vocabulary_size)

		sess.run(tf.global_variables_initializer())
		
		print "[-] Restoring model checkpoint..."
		saver = tf.train.import_meta_graph("{}.meta".format(options.checkpoint))
		saver.restore(sess, options.checkpoint)
		print "[-] Done."

        probs, state = sess.run([rnn.probs, rnn.final_state], feed_dict={ rnn.X : np.zeros([language_model.batch_size, sequence_length])})

    	while len(raw_input('Would you like to generate a sample? (Say nothing to proceed, anything to stop):')) == 0:
			print language_model.translate([int(np.searchsorted(np.cumsum(p), np.random.rand(1)*np.sum(p))) for p in probs[0:sequence_length]])