from __future__ import print_function
import json
import pandas as pd
import numpy as np
import re
import csv
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, merge
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

def _build_interaction_matrix(rows, cols, data):
    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for i in range(raw_rating_data.shape[0]):
        if raw_rating_data["rating"].loc[i] >= 4:
            mat[raw_rating_data["user id"].loc[i], raw_rating_data["item id"].loc[i]] = 1.0
    return mat.tocoo()

def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)

def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss

def build_model(num_users, num_items, latent_dim):

    positive_item_input = Input((1, ), name='positive_item_input')
    negative_item_input = Input((1, ), name='negative_item_input')

    # Shared embedding layer for positive and negative items
    item_embedding_layer = Embedding(
        num_items, latent_dim, name='item_embedding', input_length=1)

    user_input = Input((1, ), name='user_input')

    positive_item_embedding = Flatten()(item_embedding_layer(
        positive_item_input))
    negative_item_embedding = Flatten()(item_embedding_layer(
        negative_item_input))
    user_embedding = Flatten()(Embedding(
        num_users, latent_dim, name='user_embedding', input_length=1)(
            user_input))

    loss = merge(
        [positive_item_embedding, negative_item_embedding, user_embedding],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_item_input, negative_item_input, user_input],
        output=loss)
    model.compile(loss=identity_loss, optimizer=Adam())

    return model

def getdata():
    file = pd.read_json("AMAZON_FASHION_5.json", lines = True)

    df = pd.DataFrame(file)
    selected_columns = df[["reviewerID", "asin", "overall"]]
    raw_rating_data = selected_columns.copy()
    raw_rating_data = raw_rating_data.rename(columns = {'reviewerID': 'user id', 'asin': 'item id', 'overall': 'rating'})
    return raw_rating_data

def get_triplets(mat):

    return mat.row, mat.col, np.random.randint(mat.shape[1], size=len(mat.row))

raw_rating_data = getdata()
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

train, test = train_test_split(raw_rating_data, test_size=0.2, random_state=42)

def predict(model, uid, pids):

    user_vector = model.get_layer('user_embedding').get_weights()[0][uid]
    item_matrix = model.get_layer('item_embedding').get_weights()[0][pids]

    scores = (np.dot(user_vector,
                     item_matrix.T))

    return scores

def full_auc(model, ground_truth):
    """
    Measure AUC for model and ground truth on all items.
    Returns:
    - float AUC
    """

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    scores = []

    for user_id, row in enumerate(ground_truth):

        predictions = predict(model, user_id, pid_array)

        true_pids = row.indices[row.data == 1]

        grnd = np.zeros(no_items, dtype=np.int32)
        grnd[true_pids] = 1

        if len(true_pids):
            scores.append(roc_auc_score(grnd, predictions))

    return sum(scores) / len(scores)


tr_rows = train.shape[0]
tr_cols = train.shape[1]

te_rows = test.shape[0]
te_cols = test.shape[1]

tr_mat = _build_interaction_matrix(len(uniqueval),len(uniqueval2),train)
te_mat = _build_interaction_matrix(len(uniqueval),len(uniqueval2),test)

num_users, num_items = tr_mat.shape
test_uid, test_pid, test_nid = raw_rating_data.get_triplets(te_mat)

latent_dim = 100
num_epochs = 10

model = build_model(num_users, num_items, latent_dim)
print(model.summary())

# Sanity check, should be around 0.5
print('AUC before training %s' % full_auc(model, test))

for epoch in range(num_epochs):
    print('Epoch %s' % epoch)

    # Sample triplets from the training data
    uid, pid, nid = raw_rating_data.get_triplets(train)

    X = {
        'user_input': uid,
        'positive_item_input': pid,
        'negative_item_input': nid
    }

    model.fit(X,
              np.ones(len(uid)),
              batch_size=64,
              nb_epoch=1,
              verbose=0,
              shuffle=True)

    print('AUC %s' % full_auc(model, test))





