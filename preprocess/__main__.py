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


def main(label, dataset_dir, mask_type):
    train, test = get_dataset()
    train_files, test_files = train[label], test[label]
    
    
    if mask_type == 'none':
        process = process_none
    elif mask_type == 'ner':
        process = process_ner
        
    x_train = process_map(process, train_files[::-1], max_workers=10)
    x_test = process_map(process, test_files[::-1], max_workers=10)
    
    with open(os.path.join(dataset_dir, f'{label}_train.p'), 'wb') as fp:
        pickle.dump(x_train, fp)

    with open(os.path.join(dataset_dir, f'{label}_test.p'), 'wb') as fp:
        pickle.dump(x_test, fp)
        
        
if __name__ == '__main__':
    train, test = get_dataset()
    labels = set(train.keys())
    ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
    
    # filtered_labels = []
    # for k in train.keys():
    #     file_train, file_test = os.path.join('dataset_ner', f'{k}_train.p'), os.path.join('dataset_ner', f'{k}_test.p')
    #     
    #     if not os.path.exists(file_train) or not os.path.exists(file_test):
    #         filtered_labels.append(k)
    
    
    unmasked_dir = os.path.join('preprocessed_datasets', 'unmasked')
    ner_masked_dir = os.path.join('preprocessed_datasets', 'ner')
    os.makedirs(unmasked_dir, exist_ok=True)
    os.makedirs(ner_masked_dir, exist_ok=True)
    
    for label in labels:
        main(label, unmasked_dir, mask_type='none')
        main(label, ner_masked_dir, mask_type='ner')