import os
import nltk
#nltk.download('words')
#nltk.download('punkt')
import numpy as np
from nltk.tokenize import word_tokenize
from common import get_dataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

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


def load_data(dataset):
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
                x.append(tokens)
                y.append(label)
    return x, y


def bow(x, vocab_lookup=None):
    if vocab_lookup is None:
        vocab = set()
        for idx, tokens in list(enumerate(x)):
            vocab = vocab.union(set(tokens))
        
        vocab_lookup = {}
        for idx, token in enumerate(list(vocab)):
            vocab_lookup[token] = idx

    for idx, tokens in list(enumerate(x)):
        words = x[idx]
        x[idx] = np.zeros(len(vocab_lookup))
        for word in words:
            if word in vocab_lookup:
                x[idx][vocab_lookup[word]] += 1

    return np.array(x), vocab_lookup


def main():
    train, test = get_dataset()

    debug = False
    if debug:
        train_subset = {}
        for key, val in train.items():
            if len(train_subset) == 4:
                break
            train_subset[key] = val
        test_subset = {}
        for key, val in test.items():
            if len(test_subset) == 4:
                break
            test_subset[key] = val
    
        #train, = train_subset, test_subset

    x_train, y_train = load_data(train)
    x_test, y_test = load_data(test)

    x_train, vocab = bow(x_train)
    x_test, _ = bow(x_test, vocab)

    print(x_train.shape, x_test.shape)

    
    clf = LogisticRegression(random_state=0, verbose=2, max_iter=1000, n_jobs=-1)
    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))



if __name__ == '__main__':
    main()