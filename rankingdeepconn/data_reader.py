import pickle
from typing import Set, List, Dict
import sys
import numpy as np
import pandas
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import itertools
from pathlib import Path
import scipy.sparse as sp
import csv

ROOT_DIR = Path(__file__).parent.parent

def get_all_data(path="reviews.json") -> DataFrame:
    ROOT_DIR = Path(__file__).parent.parent
    return pandas.read_json(ROOT_DIR.joinpath(path), lines=True)


def get_train_dev_test_data(path="reviews.json") -> (DataFrame, DataFrame):
    all_data = get_all_data(path)
    train, test = train_test_split(all_data, test_size=0.2, random_state=42)
    return train, test

def process_raw_data(out_path="reviews.csv"):
    """
    Read raw data and remove useless columns and clear review text.
    Then save the result to file system.
    """

    print("reading raw data...")
    df = pandas.read_json("AMAZON_FASHION_5.json", lines=True)
    df = df[["reviewerID", "asin", "overall"]]
    df.columns = ["userID", "itemID", "rating"]

    df.to_csv(ROOT_DIR.joinpath(out_path), index=False ,header=False)
    print("Processed data saved.")

def _parse(data):
    """
    Parse movielens dataset lines.
    """
    csvfile = data.to_csv("prova.csv")

    with open(csvfile, 'r') as csvfile1:
        datareader = csv.reader(csvfile1)
        for line in datareader:

            if not line:
                continue

            uid, iid, rating = [str(x) for x in line.split(',')]

            yield uid, iid, rating

def get_movielens_data(train, test):
    """
    Return (train_interactions, test_interactions).
    """

    uids = set()
    iids = set()

    for uid, iid, rating in itertools.chain(_parse(train),
                                                       _parse(test)):
        uids.add(uid)
        iids.add(iid)

    rows = max(uids) + 1
    cols = max(iids) + 1

    return (_build_interaction_matrix(rows, cols, _parse(train_data)),
            _build_interaction_matrix(rows, cols, _parse(test_data)))

def _build_interaction_matrix(rows, cols, data):

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating in data:
        # Let's assume only really good things are positives
        if rating >= 4.0:
            mat[uid, iid] = 1.0

    return mat.tocoo()

def get_dense_triplets(uids, pids, nids, num_users, num_items):

    user_identity = np.identity(num_users)
    item_identity = np.identity(num_items)

    return user_identity[uids], item_identity[pids], item_identity[nids]


def get_triplets(mat):

    return mat.row, mat.col, np.random.randint(mat.shape[1], size=len(mat.row))


if __name__ == "__main__":
    process_raw_data()

    train_data, test_data = get_train_dev_test_data()

    print(train_data)
    print(test_data)

    train1, test1 = get_movielens_data(train_data, test_data)

    print(train1)
    print(test1)


