import csv
import pprint

def from_csv(csvfile):
	with open(csvfile, 'r') as f:
		reader = csv.reader(f)
		return [row for row in reader]