import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, sys
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from transformers import *
from argparse import ArgumentParser
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
    def __init__(self, config, nclasses, mlp, frozen):
        super(BertEnron, self).__init__()
        self.model_class, self.tokenizer_class, self.pretrained_weights = config
        self.lm_layer = self.model_class.from_pretrained(self.pretrained_weights)
        self.frozen = frozen
        self.main = nn.Sequential(
            nn.Linear(768, nclasses)
        ) if not mlp else nn.Sequential(
            nn.Linear(768, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, nclasses)
        )

    def forward(self, input_ids, attn_masks):
        last_hidden_states = self.lm_layer(input_ids, attention_mask=attn_masks)[0]
        cls_token = last_hidden_states[:, 0, :].detach() if self.frozen else last_hidden_states[:, 0, :]
        return self.main(cls_token)

def create_labels(dataset):
    label_to_idx, idx_to_label = {}, {}
    for idx, label in enumerate(list(dataset.keys())):
        label_to_idx[label] = idx
        idx_to_label[idx] = label
    return label_to_idx, idx_to_label


def main(config, seed=0, epochs=3, learning_rate=2e-5, batch_size=32, mlp=False, frozen=False, results_dir='results', device='cuda'):
    experiment_name = '_'.join([str(seed), config[0].__name__, config[1].__name__, config[2],
                                str(epochs), str(learning_rate), str(batch_size), str(mlp), str(frozen)])
    experiment_dir = os.path.join(results_dir, experiment_name)
    checkpoints_dir = os.path.join(results_dir, experiment_name, 'checkpoints')
    stats_filename = os.path.join(results_dir, experiment_name, 'stats.pth')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    output_file = open(os.path.join(results_dir, experiment_name, 'output.log'), 'w')
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print('Loading dataset into memory.', file=output_file, flush=True)
    train, test = get_dataset()
    label_to_idx, idx_to_label = create_labels(train)
    nclasses = len(train.keys())
    
    x_train, y_train = load_data(train, label_to_idx)
    x_test, y_test = load_data(test, label_to_idx)
    
    train = pd.DataFrame(list(zip(x_train, y_train)), columns=['Text', 'Label'])
    test = pd.DataFrame(list(zip(x_test, y_test)), columns=['Text', 'Label'])
    
    train_data = EnronDataset(df=train, config=config)
    test_data = EnronDataset(df=test, config=config)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=2, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=2)
    
    model = nn.DataParallel(BertEnron(config, nclasses, mlp, frozen)).to(device)
    
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
            
            print(f"Epoch {epoch}\tBatch {idx+1} / {len(train_dataloader)}\tLoss: {np.mean(epoch_loss):.5f}\tTrain: {correct / total:.5f}", file=output_file, flush=True)
            
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
        stats['train_accuracy'].append(train_accuracy)
        stats['test_accuracy'].append(test_accuracy)
        print(f"Epoch {stats['epoch'][-1]} Summary\tLoss: {stats['loss'][-1]:.5f}\tTrain: {stats['train_accuracy'][-1]:.5f}\tTest: {stats['test_accuracy'][-1]:.5f}", file=output_file, flush=True)
        
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'model-{epoch}.pth'))
        torch.save(stats, stats_filename)
        
    output_file.close()
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--config_id', type=int, default=0, help='pretrained model config id')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--results_dir', type=str, default='results', help='results directory')
    parser.add_argument('--frozen', type=bool, default=False, action='store_true', help='use frozen model?')
    parser.add_argument('--mlp', type=bool, default=False, action='store_true', help='use mlp classifier?')
    args = parser.parse_args()
    
    config = [(BertModel,      BertTokenizer,       'bert-base-uncased'),
              (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
              (GPT2Model,       GPT2Tokenizer,       'gpt2'),
              (CTRLModel,       CTRLTokenizer,       'ctrl'),
              (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
              (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
              (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
              (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
              (RobertaModel,    RobertaTokenizer,    'roberta-base'),
              (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base')][args.config_id]
    
    
    main(config, seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, mlp=args.mlp, frozen=args.frozen, results_dir=args.results_dir)
