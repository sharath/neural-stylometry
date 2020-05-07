import numpy as np 
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re 
from datetime import datetime, date, time, timedelta
from collections import defaultdict
import os, sys
from common import get_dataset


nltk.download('punkt')
nltk.download('vader_lexicon')


exclude_headers = [
    "Message-ID:",
    "From:",
    "To:",
    "Subject:",
    "Mime-Version:",
    "Content-Type:",
    "Content-Transfer-Encoding:",
    "X-From:",
    "X-To:",
    "X-cc: ",
    "X-bcc: ",
    "X-Folder:",
    "X-Origin:",
    "X-FileName:"
]


def load_data(dataset, text_list, time_list):
    date_tag = "Date:"
    for _, files in dataset.items():
        for file in files:
            with open(file, 'r') as fp:
                lines = fp.readlines()
                stripped = []
                for line in lines:
                    remove = False
                    if date_tag in line:
                        curr_time = line
                        remove = True 
                    for t in exclude_headers:
                        if t in line:
                            remove = True
                    if remove:
                        continue
                    stripped.append(line)
                tokens = [i for i in nltk.wordpunct_tokenize(''.join(stripped).lower())]

                text_list.append(' '.join(tokens))
                time_list.append(curr_time)


def get_sentiment(out):
	if out['pos'] > out['neg'] and out['pos'] > out['neu']:
		return 'pos'
	elif out['neg'] > out['pos'] and out['neg'] > out['neu']:
		return 'neg'
	else:
		return 'neu'


def get_time_bin(timestamp):
	match = re.search(r'\d{2}:\d{2}:\d{2}', timestamp).group().split(':')
	val = int(match[0]) * 60 * 60 + int(match[1]) * 60 + int(match[2])
	return val // (30 * 60)


def main():
	train, test = get_dataset()
	text, time = [], [] 
	load_data(train, text, time)
	load_data(test, text, time)

	sentiment_counter = {
		'pos': 0,
		'neg': 0,
		'neu': 0
	}

	pos_bins = defaultdict(int)
	neg_bins = defaultdict(int)

	for i in range(len(text)):
		sid = SentimentIntensityAnalyzer()

		seq = text[i] 
		ss = sid.polarity_scores(seq)
		sentiment = get_sentiment(ss)
		sentiment_counter[sentiment] += 1

		if sentiment == 'pos' or sentiment == 'neg':
			curr_bin = get_time_bin(time[i])
			if sentiment == 'pos':
				pos_bins[curr_bin] += 1
			else:
				neg_bins[curr_bin] += 1

	pos_vals = np.fromiter(pos_bins.values(), dtype=int)
	neg_vals = np.fromiter(neg_bins.values(), dtype=int)

	assert pos_vals.shape[0] == 24
	print(np.sum(pos_vals), np.sum(neg_vals))

	time = np.array([(datetime.combine(date.today(), base) + timedelta(hours=i)).time() for i in range(24)])

	plt.plot(time, pos_vals, label='Positive')
	plt.plot(time, neg_vals, label='Negative')
	plt.xlabel('Time during the day')
	plt.ylabel('Count')
	plt.title('Distribution of positive and negative emails throughout the day')
	plt.legend(loc='best')
	plt.savefig('bin_plot.png')
	plt.close()

	print(sentiment_counter)


if __name__ == '__main__':
	main()
