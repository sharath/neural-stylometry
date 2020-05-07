import numpy as np 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os, sys
from common import get_dataset


nltk.download('punkt')


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


def load_data(dataset):
    text, time = [], []
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
                print(curr_time, ' '.join(tokens))
                text.append(' '.join(tokens))
                time.append(curr_time)
    return text, time


train, test = get_dataset()
text, time = load_data(train)
print(len(text))
print(len(time))
