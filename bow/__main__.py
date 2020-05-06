import os
import nltk
#nltk.download('words')
import numpy as np
from nltk.tokenize import word_tokenize
from common import get_splits
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
                    if not remove:
                        stripped.append(line)
                tokens = [i for i in nltk.wordpunct_tokenize(''.join(stripped).lower())]
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

    return x, vocab_lookup


class LogRegClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogRegClassifier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(num_features, num_classes),
        )
    
    def forward(self, x):
        return self.seq(x)
    
    
    
def main():
    train, test = get_splits()
    x_train, y_train = load_data(train)
    x_test, y_test = load_data(test)
    
    x_train, vocab = bow(x_train)
    x_test, _ = bow(x_test, vocab)
    
    mappings = {j:i for (i, j) in enumerate(set(y_train))}
    
    y_train = list(map(mappings.get, y_train))
    y_test = list(map(mappings.get, y_test))
    
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    
    epochs = 100
    batch_size = len(train_dataset)//4
    learning_rate = 1e-2
    device = 'cuda:1'
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    model = LogRegClassifier(train_dataset[0][0].shape[0], len(mappings)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)
    
    stats = {'epoch':[], 'loss':[], 'train_accuracy':[], 'test_accuracy':[]}
    for epoch in range(1, epochs+1):
        epoch_loss = []
        
        model.train()
        correct, total = 0, 0
        for idx, (inpt, target) in enumerate(train_dataloader):
            inpt, target = inpt.float().to(device), target.to(device)
            model.zero_grad()
            pred = model(inpt)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            
            correct += torch.sum(pred.argmax(1) == target).item()
            total += len(target)
        train_accuracy = correct / total
        
        model.eval()
        correct, total = 0, 0
        for idx, (inpt, target) in enumerate(test_dataloader):
            inpt, target = inpt.float().to(device), target.to(device)
            pred = model(inpt)
        
            correct += torch.sum(pred.argmax(1) == target).item()
            total += len(target)
        test_accuracy = correct / total
        
        stats['epoch'].append(epoch)
        stats['loss'].append(np.mean(epoch_loss))
        stats['train_accuracy'].append(np.mean(train_accuracy))
        stats['test_accuracy'].append(np.mean(test_accuracy))
        print(f"{stats['epoch'][-1]}\t{stats['loss'][-1]:.5f}\t{stats['train_accuracy'][-1]:.5f}\t{stats['test_accuracy'][-1]:.5f}")
        
    model.save(model.state_dict(), 'bow-trained.pth')
    torch.save(stats, 'bow-stats.pth')
        
        
if __name__ == '__main__':
    main()