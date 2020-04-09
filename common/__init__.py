import os
import pickle
import numpy as np


def create_split(dataset_dir="dataset", split_file_name="splits.p", test_split=0.2, seed=0):
    np.random.seed(seed)

    labels = [f for f in os.listdir(dataset_dir)]
    email_files_train = {}
    email_files_test = {}

    for label in labels:
        email_dir = os.path.join(dataset_dir, label, "_sent_mail")
        if not os.path.isdir(email_dir):
            continue

        email_files = [os.path.join(email_dir, f) for f in os.listdir(email_dir)]
        np.random.shuffle(email_files)
        train = int((1.0 - test_split) * len(email_files))

        email_files_train[label] = email_files[:train]
        email_files_test[label] = email_files[train:]

    split_file = os.path.join(dataset_dir, split_file_name)
    with open(split_file, "wb") as fp:
        pickle.dump({"train": email_files_train, "test": email_files_test}, fp)

    return email_files_train, email_files_test


def get_dataset(dataset_dir="dataset", split_file_name="splits.p"):
    split_file = os.path.join(dataset_dir, split_file_name)
    with open(split_file, "rb") as fp:
        dataset = pickle.load(fp)
    return dataset["train"], dataset["test"]


if __name__ == "__main__":
    assert create_split() == get_dataset()
