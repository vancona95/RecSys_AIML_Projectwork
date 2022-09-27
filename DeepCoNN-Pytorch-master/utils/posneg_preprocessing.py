import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
import scipy.stats as K


def getdata():
    file = pd.read_json("AMAZON_FASHION_5.json", lines = True)

    df = pd.DataFrame(file)
    selected_columns = df[["reviewerID", "asin", "overall"]]
    raw_rating_data = selected_columns.copy()
    raw_rating_data = raw_rating_data.rename(columns = {'reviewerID': 'user id', 'asin': 'item id', 'overall': 'rating'})
    return raw_rating_data

def convertstring(raw_rating_data):
    uniqueval = raw_rating_data["user id"].unique()
    uniqueval2 = raw_rating_data["item id"].unique()
    c = 0
    d = 0
    for i in uniqueval:
        raw_rating_data["user id"] = raw_rating_data["user id"].replace([i], c)
        c = c+1

    for i in uniqueval2:
        raw_rating_data["item id"] = raw_rating_data["item id"].replace([i], d)
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

def insertmissing(lista, test, train, stringa):
    for i in lista:
        df3 = test[test[stringa] == i]
        train = pd.concat([train, df3])
    return train

def get_triplets(mat):

    return mat.row, mat.col, np.random.randint(mat.shape[1], size=len(mat.row))

def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss

def bpr_triplet_loss2(uid, pid, nid):


    # BPR loss
    loss = 1.0 - K.logistic(
        np.mean(uid * pid, axis=-1, keepdims=True) -
        np.mean(uid * nid, axis=-1, keepdims=True))

    return loss

#def _build_interaction_matrix(rows, cols, raw_rating_data):
    #mat = np.empty(rows, cols)

    #for i in range(raw_rating_data.shape[0]):
        #if raw_rating_data["rating"].loc[i] >= 4:
            #mat[raw_rating_data["user id"].loc[i], raw_rating_data["item id"].loc[i]] = 1.0
    #return mat

data = getdata()
data2 = convertstring(data)
train, test = train_test_split(data2, test_size=0.2, random_state=78)

user_list = []
item_list = []  # è nel test set e non nel training

var = checkval(train, test, 0, user_list)
var2 = checkval(train, test, 1, item_list)

print(var)
print(var2)

if len(user_list) > 0:
   train =insertmissing(user_list, test, train, 'user id')

if len(item_list) > 0:
    train = insertmissing(item_list, test, train, 'item id')

var = checkval(train, test, 0, user_list)
var2 = checkval(train, test, 1, item_list)

print(var)
print(var2)

rows = len(data["user id"].unique())
cols = len(data["item id"].unique())

print(rows)
print(cols)
#mat = np.zeros((rows, cols))

mat = sp.lil_matrix((rows, cols), dtype=np.int32)



#print(train)

for i in range(train.shape[0]):
    mat[train["user id"].iloc[i], train["item id"].iloc[i]] = 1.0

print(mat[1,24])
mat = mat.tocoo()

train_uid, train_pid, train_nid = get_triplets(mat)








