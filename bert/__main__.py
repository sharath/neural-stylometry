import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import *
import nltk
import os
from common import get_dataset


# Run this once!
nltk.download('words')


def create_labels(path='dataset'):
    label_to_idx, idx_to_label = {}, {}
    curr_idx = 0
    for sub in os.listdir(path):
        label_to_idx[sub] = curr_idx
        idx_to_label[curr_idx] = sub
        curr_idx += 1
    print(curr_idx)

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


def load_data(dataset):
    print('Loading dataset into memory.')
    x, y = [], []
    for label, files in dataset.items():
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


print('Start loading data')
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
    
    train, test = train_subset, test_subset


x_train, y_train = load_data(train)
x_test, y_test = load_data(test)

train = pd.DataFrame(list(zip(x_train, y_train)), columns=['Text', 'Label'])
test = pd.DataFrame(list(zip(x_test, y_test)), columns=['Text', 'Label'])


class EnronDataset(Dataset):

    def __init__(self, df, config, max_len=512):
        self.df = df

        self.model_class, self.tokenizer_class, self.pretrained_weights = config
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)

        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.loc[index, 'Text']
        label = self.df.loc[index, 'Label']

        tokenized_input = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_len)
        padded_input = np.array([tokenized_input + [0] * (self.max_len - len(tokenized_input))])
        mask_input = np.where(padded_input != 0, 1, 0)

        input_ids = torch.Tensor(padded_input).long()
        attn_mask = torch.Tensor(mask_input).long()

        return input_ids, attn_mask, label


class BertEnron(nn.Module):

    def __init__(self, config):
        super(BertEnron, self).__init__()
        self.model_class, self.tokenizer_class, self.pretrained_weights = config
        self.lm_layer = self.model_class.from_pretrained(self.pretrained_weights)
        self.clf_layer = nn.Linear(768, 151)

    def forward(self, input_ids, attn_masks):
        last_hidden_states = self.lm_layer(input_ids, attention_mask=attn_masks)[0]
        cls = last_hidden_states[:, 0, :]
        logits = self.clf_layer(cls)
        return logits


config = (BertModel, BertTokenizer, 'bert-base-uncased')

train_data = EnronDataset(df=train, config=config)
test_data = EnronDataset(df=test, config=config)

train_loader = DataLoader(train_data, batch_size=8, num_workers=2)
test_loader = DataLoader(test_data, batch_size=8, num_workers=2)

enron_net = BertEnron(config=config).to(device)

loss = nn.CrossEntropyLoss()
adam = optim.Adam(enron_net.parameters(), lr=2e-5)


def train_model(model, criterion, optimizer, train_loader, epochs=20):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            input_ids, attn_masks, labels = data[0].squeeze(dim=1).to(device), data[1].squeeze(dim=1).to(device), data[2].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attn_masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.4f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


# train_model(enron_net, loss, adam, train_loader, epochs=3)

PATH = './enron_bert.pth'
# torch.save(enron_net.state_dict(), PATH)

print('Loading trained model.')
enron_net.load_state_dict(torch.load(PATH))


def evaluate(model, test_loader):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in test_loader:
            input_ids, attn_mask, labels = data
            outputs = model(input_ids, attention_mask=attn_mask)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted.data.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


accuracy = evaluate(enron_net, test_loader)
print('Accuracy of the network over the test set: %d %%' % (
    100 * accuracy))
