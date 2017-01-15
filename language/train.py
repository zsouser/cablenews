from model import create, Model
import optparse
import os.path
import tensorflow as tf
import numpy as np
from rnn import RNN
import time
from flask import Flask
app = Flask(__name__)

parser = optparse.OptionParser()
parser.add_option("-c", "--csv", dest="csv", help="CSV file for model")
parser.add_option("-l", "--limit", dest="limit", help="Limit number of CSV rows processed", type="int")
parser.add_option("-x", "--checkpoint", dest="checkpoint", help="Checkpoint file for RNN", default=None)
parser.add_option("-o", "--output", dest="output", help="Output file for RNN", default=None)
parser.add_option("-r", "--learning_rate", dest="learning_rate", default=0.001, help="Learning Rate", type="int")
parser.add_option("-n", "--num_epochs", dest="num_epochs", default=3, help="Number of Epochs", type="int")
parser.add_option("-e", "--save_every", dest="save_every", default=3, help="Save a checkpoint every N epochs", type="int")
parser.add_option("-v", "--vocabulary_size", dest="vocabulary_size", default=2000, help="Vocabulary size for model", type="int")
parser.add_option("-b", "--batch_size", dest="batch_size", default=500, help="Batch size for training", type="int")

options, args = parser.parse_args()

assert os.path.exists(options.csv), "Model not found"
language_model = create(options.csv, options.limit, options.batch_size, options.vocabulary_size)

with tf.Graph().as_default():
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	with sess.as_default():
		rnn = RNN(sequence_length=language_model.max_length - 1, 
			batch_size=language_model.batch_size,
			vocabulary_size=language_model.vocabulary_size,
			learning_rate=options.learning_rate)

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
		
		if options.checkpoint is not None and os.path.exists(options.checkpoint):
			saver = tf.train.import_meta_graph("{}.meta".format(options.checkpoint))
			saver.restore(sess, options.checkpoint)

		for epoch in range(options.num_epochs):
			for batch in range(language_model.num_batches):
				start = time.time()
				x, y = language_model.batch(batch)
				feed_dict = { rnn.X : np.array(x), rnn.Y : np.array(y) }

				train_loss, state, _ = sess.run([rnn.cost, rnn.final_state, rnn.train], feed_dict)
				end = time.time()
				print "%s/%s (epoch %s), train_loss = %.3f, time/batch = %.3f" % (
					(epoch * language_model.num_batches + batch,
					options.num_epochs * language_model.num_batches,
					epoch, train_loss, end - start))

				should_save = (epoch * language_model.num_batches + batch) % options.save_every == 0
				should_save |= epoch == options.num_epochs - 1 and batch == language_model.num_batches - 1
				if should_save and options.checkpoint:
					saver.save(sess, options.checkpoint)
					print "[+] Saved to %s" % (options.checkpoint)

		@app.route("/")
		def sample():
			sequence_length = language_model.max_length - 1
			probs, state = sess.run([rnn.probs, rnn.final_state], feed_dict={ rnn.X : np.zeros([language_model.batch_size, sequence_length])})
			prediction = language_model.translate(int(np.searchsorted(np.cumsum(p), np.random.rand(1)*np.sum(p))) for p in probs[:sequence_length])
			print prediction
			return prediction

		app.run()
		print "Done!"