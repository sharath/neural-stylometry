import numpy as np
import pandas as pd
import torch
from transformers import *
from sklearn.linear_model import LogisticRegression
import nltk
from tqdm import tqdm
import os
from common import get_dataset


# Run this once!
# nltk.download('words')


def create_labels(path='dataset'):
    label_to_idx, idx_to_label = {}, {}
    curr_idx = 0
    for sub in os.listdir(path):
        label_to_idx[sub] = curr_idx
        idx_to_label[curr_idx] = sub
        curr_idx += 1

    return label_to_idx, idx_to_label


label_to_idx, idx_to_label = create_labels('dataset')

exclude_headers = [
    "Message-ID:",
    "Date:",
    "From: phillip.allen@enron.com",
    "To: outlook.team@enron.com",
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


words = set(nltk.corpus.words.words())


def load_data(dataset, limit=512):
    print('Loading dataset into memory.')
    x, y = [], []
    for label, files in tqdm(dataset.items()):
        for file in files:
            with open(file, 'r') as fp:
                lines = fp.readlines()
                stripped = []
                for line in lines:
                    remove = False
                    for t in exclude_headers:
                        if t in line:
                            remove = True
                    if remove:
                        continue
                    stripped.append(line)
                tokens = [i for i in nltk.wordpunct_tokenize(''.join(stripped).lower()) if i in words and '@enron.com' not in i]
                x.append(' '.join(tokens))
                y.append(label_to_idx[label])
    return x, y


train, test = get_dataset()
x_train, y_train = load_data(train)
x_test, y_test = load_data(test)

train = pd.DataFrame(list(zip(x_train, y_train)), columns=['Text', 'Label'])
test = pd.DataFrame(list(zip(x_test, y_test)), columns=['Text', 'Label'])


model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


def get_features(df, model, tokenizer, max_len=512):
    tokenized_train = df['Text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_len)))
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized_train.values])
    attn_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)
    attn_mask = torch.tensor(attn_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attn_mask)

    features = last_hidden_states[0][:, 0, :].numpy()
    return features


train_features = get_features(train, model, tokenizer)
test_features = get_features(test, model, tokenizer)

train_labels = train['Label']
test_labels = test['Label']

clf = LogisticRegression()
clf.fit(train_features, y_train)
print(clf.score(train_features, train_labels))
print(clf.score(test_features, test_labels))
