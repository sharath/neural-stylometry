import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
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
	texts, timestamps = [], [] 
	load_data(train, texts, timestamps)
	load_data(test, texts, timestamps)

	sentiment_counter = {
		'pos': 0,
		'neg': 0,
		'neu': 0
	}

	pos_bins, neg_bins = {}, {}
	for t in range(48):
		pos_bins[t] = 0
		neg_bins[t] = 0

	for i in tqdm(range(len(texts))):
		sid = SentimentIntensityAnalyzer()

		seq = texts[i] 
		ss = sid.polarity_scores(seq)
		sentiment = get_sentiment(ss)
		sentiment_counter[sentiment] += 1

		if sentiment == 'pos' or sentiment == 'neg':
			curr_bin = get_time_bin(timestamps[i])
			if sentiment == 'pos':
				pos_bins[curr_bin] += 1
			else:
				neg_bins[curr_bin] += 1

	pos_vals = np.fromiter(pos_bins.values(), dtype=int)
	neg_vals = np.fromiter(neg_bins.values(), dtype=int)

	print(pos_vals.shape, neg_vals.shape)
	print(np.sum(pos_vals), np.sum(neg_vals))

	base = time(0, 0, 0)
	hour_list = np.array([datetime.combine(date.today(), base) + timedelta(minutes=i*30) for i in range(48)])
	

	plt.plot(hour_list, pos_vals, label='Positive')
	plt.plot(hour_list, neg_vals, label='Negative')

	xformatter = mdates.DateFormatter('%H:%M')
	plt.gcf().axes[0].xaxis.set_major_formatter(xformatter)

	plt.xlabel('Time during the day')
	plt.ylabel('Count')
	plt.title('Distribution of positive and negative emails throughout the day')
	plt.legend(loc='best')
	plt.savefig('bin_plot.png')
	plt.close()

	print(sentiment_counter)


if __name__ == '__main__':
	main()
