import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from transformers import *
from common import get_dataset

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


def load_data(dataset, label_to_idx):
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
                tokens = [i for i in nltk.wordpunct_tokenize(''.join(stripped).lower())]
                x.append(' '.join(tokens))
                y.append(label_to_idx[label])
    return x, y

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

    def __init__(self, config, nclasses):
        super(BertEnron, self).__init__()
        self.model_class, self.tokenizer_class, self.pretrained_weights = config
        self.lm_layer = self.model_class.from_pretrained(self.pretrained_weights)
        self.clf_layer = nn.Linear(768, nclasses)

    def forward(self, input_ids, attn_masks):
        last_hidden_states = self.lm_layer(input_ids, attention_mask=attn_masks)[0]
        cls = last_hidden_states[:, 0, :]
        logits = self.clf_layer(cls)
        return logits

    
#def create_labels(path='dataset'):
#    label_to_idx, idx_to_label = {}, {}
#    curr_idx = 0
#    for sub in os.listdir(path):
#        label_to_idx[sub] = curr_idx
#        idx_to_label[curr_idx] = sub
#        curr_idx += 1
#
#    return label_to_idx, idx_to_label
    
def create_labels(dataset):
    label_to_idx, idx_to_label = {}, {}
    for idx, label in enumerate(list(dataset.keys())):
        label_to_idx[label] = idx
        idx_to_label[idx] = label
    return label_to_idx, idx_to_label
    
def main():
    device = 'cuda'
    epochs = 3
    learning_rate = 2e-5
    batch_size = 32
    
    print('Start loading data')
    
    train, test = get_dataset()
    label_to_idx, idx_to_label = create_labels(train)
    nclasses = len(train.keys())
    
    x_train, y_train = load_data(train, label_to_idx)
    x_test, y_test = load_data(test, label_to_idx)
    
    train = pd.DataFrame(list(zip(x_train, y_train)), columns=['Text', 'Label'])
    test = pd.DataFrame(list(zip(x_test, y_test)), columns=['Text', 'Label'])
    
    config = (BertModel, BertTokenizer, 'bert-base-uncased')
    
    train_data = EnronDataset(df=train, config=config)
    test_data = EnronDataset(df=test, config=config)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=2)
    
    model = nn.DataParallel(BertEnron(config=config, nclasses=nclasses)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    stats = {'epoch':[], 'loss':[], 'train_accuracy':[], 'test_accuracy':[]}
    for epoch in range(1, epochs+1):
        epoch_loss = []
        
        model.train()
        correct, total = 0, 0
        for idx, (input_ids, attn_masks, target) in enumerate(train_dataloader):
            input_ids = input_ids.squeeze(dim=1).to(device)
            attn_masks = attn_masks.squeeze(dim=1).to(device)
            target = target.to(device)
            
            model.zero_grad()
            pred = model(input_ids, attn_masks)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            
            correct += torch.sum(pred.argmax(1) == target).item()
            total += len(target)
            
        train_accuracy = correct / total
        
        model.eval()
        correct, total = 0, 0
        for idx, (input_ids, attn_masks, target) in enumerate(test_dataloader):
            input_ids = input_ids.squeeze(dim=1).to(device)
            attn_masks = attn_masks.squeeze(dim=1).to(device)
            target = target.to(device)
            pred = model(input_ids, attn_masks)
            correct += torch.sum(pred.argmax(1) == target).item()
            total += len(target)
            
        test_accuracy = correct / total
        
        stats['epoch'].append(epoch)
        stats['loss'].append(np.mean(epoch_loss))
        stats['train_accuracy'].append(np.mean(train_accuracy))
        stats['test_accuracy'].append(np.mean(test_accuracy))
        print(f"{stats['epoch'][-1]}\t{stats['loss'][-1]:.5f}\t{stats['train_accuracy'][-1]:.5f}\t{stats['test_accuracy'][-1]:.5f}")
        
        torch.save(model.state_dict(), f'bert-{epoch}.pth')
        torch.save(stats, f'bert-stats.pth')
    
    
if __name__ == '__main__':
    main()