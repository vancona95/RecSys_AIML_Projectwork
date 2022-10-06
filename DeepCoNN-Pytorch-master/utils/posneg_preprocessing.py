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
import warnings

warnings.filterwarnings('ignore')

#legge il dataset con le recensioni già processate e lo converte in df
def getdata():
    file = pd.read_json("reviews.json", lines = True)

    df = pd.DataFrame(file)
    #selected_columns = df[["reviewerID", "asin", "overall", "reviewText"]]
    #raw_rating_data = selected_columns.copy()
    #raw_rating_data = raw_rating_data.rename(columns = {'reviewerID': 'userID', 'asin': 'itemID', 'overall': 'rating', 'reviewText': 'review'})
    return df

#converte gli user e gli item ids da stringhe alfanumeriche a interi crescenti (mapping)
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

#controlla che gli user contenuti nel test set siano presenti anche nel train set
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

#rimuove gli item presenti nel test set che non sono presenti nel training set
def removemissing(lista, test, stringa):
    for i in lista:
        test.drop(test[test[stringa] == i].index, inplace=True)
    return test

#estrae le triple userid, item positivi e item negativi
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

#effettua lo split del dataset in training e test set (80 - 20) su base utente
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

    review_by_user = dict(list(data[["posID", "posReview"]].groupby(data["userID"])))
    review_by_positem = dict(list(data[["userID", "posReview"]].groupby(data["posID"])))
    review_by_negitem = dict(list(data[["userID", "negReview"]].groupby(data["negID"])))

    return review_by_user, review_by_positem, review_by_negitem

#-----------------PREPROCESSING DATASET ----------------------------------

dr.process_raw_data() #preprocessing sulle recensioni
data = getdata()
data2 = convertids(data)

train, test = split(data2)
item_list = []

var2 = checkval(train, test, 1, item_list)

#rimuoviamo gli item non presenti nel train set
if len(item_list) > 0:
    test = removemissing(item_list, test, 'itemID')
var2 = checkval(train, test, 1, item_list)

#-------------------------------------------------------------------------
#----------------TRIPLE EXTRACTION----------------------------------------
rows = len(data["userID"].unique())
cols = len(data["itemID"].unique())

#creiamo una matrice sparsa avente dimensioni userID (righe) e itemID (colonne). mat(x,y) == 1 -> l'item y è positivo per l'utente x.  mat(x,y) == 0 -> l'item y è negativo per l'utente x.
mat = sp.lil_matrix((rows, cols), dtype=np.int32)

for i in range(train.shape[0]):
    mat[train["userID"].iloc[i], train["itemID"].iloc[i]] = 1.0

mat = mat.tocoo()

#generiamo il word_vec per il modello word2vec e convertiamo ogni review del training set in una lista di indici interi
word_vec = dr.get_word_vec()
train["review"] = train["review"].apply(w2vh.review2wid, args=[word_vec])

train_uid, train_pid, train_nid = get_triplets(mat, train)

#generiamo un nuovo train set avente le triple estratte
new_train = pd.DataFrame(columns= ["userID", "posID", "negID", "userReview", "posReview", "negReview"])
new_train["userID"] = train_uid
new_train["posID"] = train_pid
new_train["negID"] = train_nid

#inseriamo nel dataframe appena creato le recensioni relative a tutti gli utenti, agli item positivi e agli item negativi
for i in range(new_train.shape[0]):
    reviews_df = train.loc[train["userID"] == new_train["userID"].iloc[i], "review"]
    reviews_list = reviews_df.to_list()
    new_train["userReview"].iloc[i] = reviews_list


for i in range(new_train.shape[0]):
    reviews_df = train.loc[(train["itemID"] == new_train["posID"].iloc[i]) & (train["userID"] == new_train["userID"].iloc[i]), "review"]
    reviews_list = reviews_df.to_list()
    new_train["posReview"].iloc[i] = reviews_list


for i in range(new_train.shape[0]):
    reviews_df = train.loc[train["itemID"] == new_train["negID"].iloc[i], "review"]
    reviews_list = reviews_df.to_list() #reviews_list contiene tutte le recensioni per ciascun item negativo
    new_train["negReview"].iloc[i] = reviews_list[0] #con [0] proviamo a prendere una sola recensione per ciascun item negativo

#with pd.option_context('display.max_columns', None):  # more options can be specified also    print(df)
    #print(new_train)

#dr.save_embedding_weights(word_vec)

#raggruppiamo le recensioni in new_train in base a userID, posID e negID
review_by_user, review_by_positem, review_by_negitem = get_reviews_in_idx2(new_train, word_vec)

print(review_by_user, end="\n")
print(review_by_positem, end="\n")
print(review_by_negitem, end="\n")
#pickle.dump(user_review, open(ROOT_DIR.joinpath("data/user_review_word_idx.p"), "wb"))
#pickle.dump(item_review, open(ROOT_DIR.joinpath("data/item_review_word_idx.p"), "wb"))











