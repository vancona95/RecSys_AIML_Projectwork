import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
import scipy.stats as K
import data_reader as dr
import word2vec_hepler as w2vh
from pandas import DataFrame
import pickle
from typing import Set, List, Dict


def getdata():
    file = pd.read_json("reviews.json", lines = True)

    df = pd.DataFrame(file)
    #selected_columns = df[["reviewerID", "asin", "overall", "reviewText"]]
    #raw_rating_data = selected_columns.copy()
    #raw_rating_data = raw_rating_data.rename(columns = {'reviewerID': 'userID', 'asin': 'itemID', 'overall': 'rating', 'reviewText': 'review'})
    return df


def convertids(raw_rating_data):
    uniqueval = raw_rating_data["userID"].unique()
    uniqueval2 = raw_rating_data["itemID"].unique()
    c = 0
    d = 0
    for i in uniqueval:
        raw_rating_data["userID"] = raw_rating_data["userID"].replace([i], c)
        c = c+1

    for i in uniqueval2:
        raw_rating_data["itemID"] = raw_rating_data["itemID"].replace([i], d)
        d = d+1
    return raw_rating_data


def checkval(train, test, index, lista):
    bul = True
    df2 = test.iloc[:, index]
    unici = df2.unique()
    col_one_list = train.iloc[:, index].to_list()
    for i in unici:
        if not i in col_one_list:
            lista.append(i)
            bul = False
    return bul


def removemissing(lista, test, stringa):
    for i in lista:
        test.drop(test[test[stringa] == i].index, inplace=True)
    return test


def get_triplets(mat, train):
    vec = []
    for user in mat.row:
        positem_series = train.loc[train["userID"] == user, "itemID"]
        positem_list = positem_series.to_list()
        matcol = mat.col
        matcol = matcol.flatten()
        x = np.random.choice(matcol)
        while x in positem_list:
            x = np.random.choice(matcol)
        vec.append(x)
    return mat.row, mat.col, vec

def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss

def split(data):
    uniqueval = data["userID"].unique()
    train = pd.DataFrame()
    test = pd.DataFrame()
    for i in uniqueval:
        df = data.loc[data["userID"] == i]
        train2, test2 = train_test_split(df, test_size=0.2, random_state=54)
        train = pd.concat([train, train2])
        test = pd.concat([test, test2])
    return train, test

def get_reviews_in_idx2(data: DataFrame, word_vec) -> (Dict[str, DataFrame], Dict[str, DataFrame]):
    """
    1. Group review by user and item.
    2. Convert word into word idx.
    :return The dictionary from userID/itemID to review text in word idx with itemID/userID.
    """

    #data["review"] = data["review"].apply(w2vh.review2wid, args=[word_vec])

    review_by_posuser = dict(list(data[["posID", "userReview"]].groupby(data["userID"])))
    review_by_neguser = dict(list(data[["negID", "userReview"]].groupby(data["userID"])))
    review_by_positem = dict(list(data[["userID", "posReview"]].groupby(data["posID"])))
    review_by_negitem = dict(list(data[["userID", "negReview"]].groupby(data["negID"])))

    return review_by_posuser, review_by_neguser, review_by_positem, review_by_negitem

dr.process_raw_data() #preprocessing sulle recensioni
data = getdata()
data2 = convertids(data)
train, test = split(data2)

item_list = []  # è nel test set e non nel training

var2 = checkval(train, test, 1, item_list)

if len(item_list) > 0:
    test = removemissing(item_list, test, 'itemID')

var2 = checkval(train, test, 1, item_list)

rows = len(data["userID"].unique())
cols = len(data["itemID"].unique())

mat = sp.lil_matrix((rows, cols), dtype=np.int32)

for i in range(train.shape[0]):
    mat[train["userID"].iloc[i], train["itemID"].iloc[i]] = 1.0

print(mat[2,1])
mat = mat.tocoo()

word_vec = dr.get_word_vec()
train["review"] = train["review"].apply(w2vh.review2wid, args=[word_vec])


train_uid, train_pid, train_nid = get_triplets(mat, train)

new_train = pd.DataFrame(columns= ["userID", "posID", "negID", "userReview", "posReview", "negReview"])
new_train["userID"] = train_uid
new_train["posID"] = train_pid
new_train["negID"] = train_nid

for i in range(new_train.shape[0]):
    reviews_df = train.loc[train["userID"] == new_train["userID"].iloc[i], "review"]
    reviews_list = reviews_df.to_list()
    new_train["userReview"].iloc[i] = reviews_list


for i in range(new_train.shape[0]):
    reviews_df = train.loc[train["itemID"] == new_train["posID"].iloc[i], "review"]
    reviews_list = reviews_df.to_list()
    new_train["posReview"].iloc[i] = reviews_list


for i in range(new_train.shape[0]):
    reviews_df = train.loc[train["itemID"] == new_train["negID"].iloc[i], "review"]
    reviews_list = reviews_df.to_list()
    new_train["negReview"].iloc[i] = reviews_list




print(type(new_train["userReview"].iloc[0]))

#with pd.option_context('display.max_columns', None):  # more options can be specified also    print(df)
    #print(new_train)

#dr.save_embedding_weights(word_vec)

review_by_posuser, review_by_neguser, review_by_positem, review_by_negitem = get_reviews_in_idx2(new_train, word_vec)

print(review_by_posuser)
#pickle.dump(user_review, open(ROOT_DIR.joinpath("data/user_review_word_idx.p"), "wb"))
#pickle.dump(item_review, open(ROOT_DIR.joinpath("data/item_review_word_idx.p"), "wb"))











