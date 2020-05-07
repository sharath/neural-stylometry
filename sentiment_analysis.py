import numpy as np 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re 
from datetime import datetime
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

	for i in range(len(text)):
		sid = SentimentIntensityAnalyzer()
		match = re.search(r'\d{2}:\d{2}:\d{2}', time[i])
		# print(match) 
		seq = text[i] 
		ss = sid.polarity_scores(seq)
		sent = get_sentiment(ss)
		sentiment_counter[sent] += 1
	print(sentiment_counter)


if __name__ == '__main__':
	main()
