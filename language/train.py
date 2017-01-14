from model import restore, Model
import optparse
import os.path
import tensorflow as tf
import numpy as np
from rnn import RNN
import time

parser = optparse.OptionParser()
parser.add_option("-m", "--model", dest="model", help="Pickle file for existing model")
parser.add_option("-c", "--checkpoint", dest="checkpoint", help="Checkpoint file for RNN", default=None)
parser.add_option("-o", "--output", dest="output", help="Path for checkpoint output")
parser.add_option("-l", "--learning_rate", dest="learning_rate", default=0.001, help="Learning Rate", type="int")
parser.add_option("-e", "--num_epochs", dest="num_epochs", default=5, help="Number of Epochs", type="int")
parser.add_option("-s", "--save_every", dest="save_every", default=2, help="Save a checkpoint every N epochs", type="int")

options, args = parser.parse_args()
model = None

assert os.path.exists(options.model), "Model not found"
language_model = restore(options.model)

with tf.Graph().as_default():
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	with sess.as_default():
		rnn = RNN(sequence_length=language_model.max_length - 1, 
			batch_size=language_model.batch_size,
			vocabulary_size=language_model.vocabulary_size,
			learning_rate=options.learning_rate)

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
		
		if os.path.exists(options.checkpoint):
			saver = tf.train.import_meta_graph("{}.meta".format(options.checkpoint))
			saver.restore(sess, options.checkpoint)

		for epoch in range(options.num_epochs):
			for batch in range(language_model.num_batches):
				start = time.time()
				x, y = language_model.batch(batch)
				feed_dict = { rnn.X : x, rnn.Y : y }

				train_loss, state, _ = sess.run([rnn.cost, rnn.final_state, rnn.train], feed_dict)
				end = time.time()
				print "%s/%s (epoch %s), train_loss = %.3f, time/batch = %.3f" % (
					(epoch * language_model.num_batches + batch,
					options.num_epochs * language_model.num_batches,
					epoch, train_loss, end - start))

				should_save = (epoch * language_model.num_batches + batch) % options.save_every == 0
				should_save |= epoch == options.num_epochs - 1 and batch == language_model.num_batches - 1
				if should_save and options.output:
					saver.save(sess, options.output)
					print "[+] Saved to %s" % (options.output)
		print "Done!"