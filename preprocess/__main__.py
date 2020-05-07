import os
import nltk
import pickle
from common import get_dataset
from nltk.parse import CoreNLPParser
from tqdm.contrib.concurrent import process_map

def create_labels(dataset):
    label_to_idx, idx_to_label = {}, {}
    for idx, label in enumerate(list(dataset.keys())):
        label_to_idx[label] = idx
        idx_to_label[idx] = label
    return label_to_idx, idx_to_label
    
    
exclude_headers = set([
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
])


ner_tagger = None
pos_tagger = None

def process_none(file):
    with open(file, 'r') as fp:
        lines = fp.readlines()
        stripped = []
        for line in lines:
            remove = False
            for t in exclude_headers:
                if t in line:
                    remove = True
                    break
            if remove:
                continue
            stripped.append(line)
        
        tokens = [i for i in nltk.wordpunct_tokenize(''.join(stripped))][:1024]
    return tokens

def process_posuh(file):
    global pos_tagger
    with open(file, 'r') as fp:
        lines = fp.readlines()
        stripped = []
        for line in lines:
            remove = False
            for t in exclude_headers:
                if t in line:
                    remove = True
                    break
            if remove:
                continue
            stripped.append(line)
        
        tokens = [i for i in nltk.wordpunct_tokenize(''.join(stripped))][:1024]
        tagged = pos_tagger.tag(tokens)
        mapping = lambda x, y : x.lower() if y != 'UH' else '[POSUH]'
        tokens = [mapping(k, v) for (k, v) in tagged]
    return tokens

def process_ner(file):
    global ner_tagger
    with open(file, 'r') as fp:
        lines = fp.readlines()
        stripped = []
        for line in lines:
            remove = False
            for t in exclude_headers:
                if t in line:
                    remove = True
                    break
            if remove:
                continue
            stripped.append(line)
        
        tokens = [i for i in nltk.wordpunct_tokenize(''.join(stripped))][:1024]
        tagged = ner_tagger.tag(tokens)
        mapping = lambda x, y : x.lower() if y == 'O' else '[NER]'
        tokens = [mapping(k, v) for (k, v) in tagged]
    return tokens

def process_posppn(file):
    global pos_tagger
    masked = set(['NNP', 'NNPS'])
    with open(file, 'r') as fp:
        lines = fp.readlines()
        stripped = []
        for line in lines:
            remove = False
            for t in exclude_headers:
                if t in line:
                    remove = True
                    break
            if remove:
                continue
            stripped.append(line)
        
        tokens = [i for i in nltk.wordpunct_tokenize(''.join(stripped))][:1024]
        tagged = pos_tagger.tag(tokens)
        mapping = lambda x, y : x.lower() if y not in masked else '[POSPPN]'
        tokens = [mapping(k, v) for (k, v) in tagged]
    return tokens


def main(label, dataset_dir, process):
    train, test = get_dataset()
    train_files, test_files = train[label], test[label]
            
    x_train = process_map(process, train_files[::-1], max_workers=10)
    x_test = process_map(process, test_files[::-1], max_workers=10)
    
    with open(os.path.join(dataset_dir, f'{label}_train.p'), 'wb') as fp:
        pickle.dump(x_train, fp)

    with open(os.path.join(dataset_dir, f'{label}_test.p'), 'wb') as fp:
        pickle.dump(x_test, fp)
        
        
if __name__ == '__main__':
    train, test = get_dataset()
    labels = set(train.keys())
    
    unmasked_dir = os.path.join('preprocessed_datasets', 'unmasked')
    os.makedirs(unmasked_dir, exist_ok=True)
    
    ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
    ner_masked_dir = os.path.join('preprocessed_datasets', 'ner')
    os.makedirs(ner_masked_dir, exist_ok=True)
    
    pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
    
    posuh_masked_dir = os.path.join('preprocessed_datasets', 'posuh')
    os.makedirs(posuh_masked_dir, exist_ok=True)
    
    posppn_masked_dir = os.path.join('preprocessed_datasets', 'posppn')
    os.makedirs(posppn_masked_dir, exist_ok=True)
    
    for label in labels:
        main(label, unmasked_dir, process_none)
        main(label, ner_masked_dir, process_ner)
        main(label, posuh_masked_dir, process_posuh)
        main(label, posppn_masked_dir, process_posppn)