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


def create_labels(dataset):
    label_to_idx, idx_to_label = {}, {}
    for idx, label in enumerate(list(dataset.keys())):
        label_to_idx[label] = idx
        idx_to_label[idx] = label
    return label_to_idx, idx_to_label


exclude_headers = [
    "Message-ID:",
    "Date:",
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


words = set(nltk.corpus.words.words())


def load_data(dataset, label_to_idx_dict):
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
                y.append(label_to_idx_dict[label])
    return x, y


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


print('Start loading data')
train, test = get_dataset()
label_to_idx, idx_to_label = create_labels(train)
num_labels = len(train.keys())
print('Number of labels: {}'.format(num_labels))

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


x_train, y_train = load_data(train, label_to_idx)
x_test, y_test = load_data(test, label_to_idx)

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

    def __getitem__(self, idx):
        text = self.df.loc[idx, 'Text']
        label = self.df.loc[idx, 'Label']

        tokenized_input = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_len)
        padded_input = np.array([tokenized_input + [0] * (self.max_len - len(tokenized_input))])
        mask_input = np.where(padded_input != 0, 1, 0)

        input_ids = torch.Tensor(padded_input).long()
        attn_mask = torch.Tensor(mask_input).long()

        return input_ids, attn_mask, label


class BertEnron(nn.Module):

    def __init__(self, num_classes, config):
        super(BertEnron, self).__init__()
        self.model_class, self.tokenizer_class, self.pretrained_weights = config
        self.lm_layer = self.model_class.from_pretrained(self.pretrained_weights)
        self.fc1 = nn.Linear(768, 100)
        self.bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, input_ids, attn_masks):
        last_hidden_states = self.lm_layer(input_ids, attention_mask=attn_masks)[0]
        cls_token = last_hidden_states[:, 0, :]

        x = self.bn(F.relu(self.fc1(cls_token)))
        logits = self.fc2(x)
        return logits


config = (BertModel, BertTokenizer, 'bert-base-uncased')

train_data = EnronDataset(df=train, config=config)
test_data = EnronDataset(df=test, config=config)

train_loader = DataLoader(train_data, batch_size=8, num_workers=2, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, num_workers=2, shuffle=True)

enron_net = BertEnron(num_classes=num_labels, config=config).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(enron_net.parameters(), lr=2e-5)


def train_model(model, criterion, optimizer, loader, epochs=3, print_every=1000):
    batch_history, epoch_history = [], []
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(loader):
            input_ids, attn_masks, labels = data[0].squeeze(dim=1).to(device), data[1].squeeze(dim=1).to(device), data[2].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attn_masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_history.append(loss.item())
            epoch_loss += loss.item()

            running_loss += loss.item()
            if i % print_every == print_every - 1:
                print('[%d, %5d] loss: %.4f' %
                      (epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0

        epoch_history.append(epoch_loss)

    return np.array(batch_history), np.array(epoch_history)


batch_losses, epoch_losses = train_model(enron_net, criterion, optimizer, train_loader, epochs=3)
np.save(file='bert_batch_loss', arr=batch_losses)
np.save(file='bert_epoch_loss', arr=epoch_losses)

PATH = './bert_2_layer.pth'
torch.save(enron_net.state_dict(), PATH)

# print('Loading trained model.')
# enron_net.load_state_dict(torch.load(PATH))


def evaluate(model, loader):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(loader):
            input_ids, attn_masks, labels = data[0].squeeze(dim=1).to(device), data[1].squeeze(dim=1).to(device), data[2].to(device)
            outputs = model(input_ids, attn_masks)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Correctly predicted: {} out of {}.'.format(correct, total))
    return correct / total


accuracy = evaluate(enron_net, test_loader)
print('Accuracy of the network over the test set: %d %%' % (100 * accuracy))
