from utils import from_csv
import os.path
import optparse
import pickle
from model import Model

parser = optparse.OptionParser()
parser.add_option("-p", "--pickle", dest="pickle", help="Pickle file for existing model")
parser.add_option("-c", "--csv", dest="filename", help="CSV Input for Language Model")
parser.add_option("-o", "--output", dest="output", help="Path for serialized output")
parser.add_option("-r", "--rnn", dest="rnn", default=False, help="Build RNN to this path")

options, args = parser.parse_args()

assert os.path.exists(options.filename), "No input file found"

data = from_csv(options.filename, lambda x: x[1] + ": " + x[2])
model = None

if options.pickle:
	assert os.path.exists(options.pickle), "No input file found"
	model = Model(pickle=options.pickle)
else:
	model = Model(data)

print model.lookup
print model.index

if options.rnn:
	rnn = RNN(model)

if options.output:
	pickle.dump(model, open("./models/" + options.output + ".pickle", 'w'))