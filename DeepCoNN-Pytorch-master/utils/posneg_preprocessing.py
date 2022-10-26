import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np

import data_reader as dr
import word2vec_hepler as w2vh
from pandas import DataFrame
import pickle
from typing import Set, List, Dict
import warnings
import itertools
import torch.nn as nn

import math
import time
from itertools import chain
from typing import Dict, List

import torch
from pandas import DataFrame
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from BaseModel import BaseModel, BaseConfig
from log_hepler import logger, add_log_file, remove_log_file
from path_helper import ROOT_DIR
from word2vec_hepler import PAD_WORD_ID
from word2vec_hepler import WORD_EMBEDDING_SIZE, load_embedding_weights
from DeepCoNN import DeepCoNNConfig, DeepCoNN
import train_helper as th

warnings.filterwarnings('ignore')


# legge il dataset con le recensioni già processate e lo converte in df
def getdata():
    file = pd.read_json("reviews.json", lines=True)

    df = pd.DataFrame(file)
    # selected_columns = df[["reviewerID", "asin", "overall", "reviewText"]]
    # raw_rating_data = selected_columns.copy()
    # raw_rating_data = raw_rating_data.rename(columns = {'reviewerID': 'userID', 'asin': 'itemID', 'overall': 'rating', 'reviewText': 'review'})
    return df


# converte gli user e gli item ids da stringhe alfanumeriche a interi crescenti (mapping)
def convertids(raw_rating_data):
    uniqueval = raw_rating_data["userID"].unique()
    uniqueval2 = raw_rating_data["itemID"].unique()
    c = 0
    d = 0
    for i in uniqueval:
        raw_rating_data["userID"] = raw_rating_data["userID"].replace([i], c)
        c = c + 1

    for i in uniqueval2:
        raw_rating_data["itemID"] = raw_rating_data["itemID"].replace([i], d)
        d = d + 1
    return raw_rating_data


# controlla che gli user contenuti nel test set siano presenti anche nel train set
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


# rimuove gli item presenti nel test set che non sono presenti nel training set
def removemissing(lista, test, stringa):
    for i in lista:
        test.drop(test[test[stringa] == i].index, inplace=True)
    return test


# estrae le triple userid, item positivi e item negativi
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

    sum1 = user_latent * positive_item_latent
    sum2 = user_latent * negative_item_latent

    input2 = torch.sum(sum1) - torch.sum(sum2)

    # BPR loss
    loss = 1.0 - torch.nn.Sigmoid(input2)

    return loss


# effettua lo split del dataset in training e test set (80 - 20) su base utente
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

    # data["review"] = data["review"].apply(w2vh.review2wid, args=[word_vec])

    review_by_user = dict(list(data[["posID", "posReview"]].groupby(data["userID"])))
    review_by_positem = dict(list(data[["userID", "posReview"]].groupby(data["posID"])))
    review_by_negitem = dict(list(data[["userID", "negReview"]].groupby(data["negID"])))

    return review_by_user, review_by_positem, review_by_negitem


def get_review_dict2():
    user_review = pickle.load(open(ROOT_DIR.joinpath("data/user_review_word_idx2.p"), "rb"))
    item_review = pickle.load(open(ROOT_DIR.joinpath("data/item_review_word_idx2.p"), "rb"))
    return user_review, item_review


def get_data_train_test_preprocessed():
    dr.process_raw_data()  # preprocessing sulle recensioni
    data = getdata()
    data2 = convertids(data)

    train, test = split(data2)
    test.to_csv("test1.csv")

    return data, train, test


def main_preprocessing():
    # -----------------PREPROCESSING DATASET ----------------------------------
    data, train, test = get_data_train_test_preprocessed()
    item_list = []

    var2 = checkval(train, test, 1, item_list)

    # rimuoviamo gli item non presenti nel train set
    if len(item_list) > 0:
        test = removemissing(item_list, test, 'itemID')
    var2 = checkval(train, test, 1, item_list)

    # -------------------------------------------------------------------------
    # ----------------TRIPLE EXTRACTION----------------------------------------
    rows = len(data["userID"].unique())
    cols = len(data["itemID"].unique())

    # creiamo una matrice sparsa avente dimensioni userID (righe) e itemID (colonne). mat(x,y) == 1 -> l'item y è positivo per l'utente x.  mat(x,y) == 0 -> l'item y è negativo per l'utente x.
    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for i in range(train.shape[0]):
        mat[train["userID"].iloc[i], train["itemID"].iloc[i]] = 1.0

    mat = mat.tocoo()

    # generiamo il word_vec per il modello word2vec e convertiamo ogni review del training set in una lista di indici interi
    word_vec = dr.get_word_vec()
    train["review"] = train["review"].apply(w2vh.review2wid, args=[word_vec])

    train_uid, train_pid, train_nid = get_triplets(mat, train)

    # generiamo un nuovo train set avente le triple estratte
    new_train = pd.DataFrame(columns=["userID", "posID", "negID", "userReview", "posReview", "negReview"])
    new_train["userID"] = train_uid
    new_train["posID"] = train_pid
    new_train["negID"] = train_nid

    # inseriamo nel dataframe appena creato le recensioni relative a tutti gli utenti, agli item positivi e agli item negativi
    for i in range(new_train.shape[0]):
        reviews_df = train.loc[train["userID"] == new_train["userID"].iloc[i], "review"]
        reviews_list = reviews_df.to_list()
        new_train["userReview"].iloc[i] = reviews_list
        new_train["userReview"].iloc[i] = list(itertools.chain.from_iterable(
            new_train["userReview"].iloc[i]))  # concateniamo le diverse recensioni in un'unica lista di parole

    for i in range(new_train.shape[0]):
        reviews_df = train.loc[(train["itemID"] == new_train["posID"].iloc[i]) & (
                train["userID"] == new_train["userID"].iloc[i]), "review"]
        reviews_list = reviews_df.to_list()
        new_train["posReview"].iloc[i] = reviews_list
        new_train["posReview"].iloc[i] = list(itertools.chain.from_iterable(
            new_train["posReview"].iloc[i]))  # concateniamo le diverse recensioni in un'unica lista di parole

    for i in range(new_train.shape[0]):
        reviews_df = train.loc[train["itemID"] == new_train["negID"].iloc[i], "review"]
        reviews_list = reviews_df.to_list()  # reviews_list contiene tutte le recensioni per ciascun item negativo
        new_train["negReview"].iloc[i] = reviews_list
        new_train["negReview"].iloc[i] = list(itertools.chain.from_iterable(
            new_train["negReview"].iloc[i]))  # concateniamo le diverse recensioni in un'unica lista di parole

    # with pd.option_context('display.max_columns', None):  # more options can be specified also    print(df)
    # print(new_train)

    # dr.save_embedding_weights(word_vec)

    # raggruppiamo le recensioni in new_train in base a userID, posID e negID
    review_by_user, review_by_positem, review_by_negitem = get_reviews_in_idx2(new_train, word_vec)

    # print(review_by_user) #posID #posReview
    # print("\n")
    # print(review_by_positem)
    # print("\n")
    # print(review_by_negitem)
    # print("\n")

    return new_train, test, review_by_user, review_by_positem, review_by_negitem
    # pickle.dump(user_review, open(ROOT_DIR.joinpath("data/user_review_word_idx.p"), "wb"))
    # pickle.dump(item_review, open(ROOT_DIR.joinpath("data/item_review_word_idx.p"), "wb"))


def save_model(model: torch.nn.Module, train_time: time.struct_time):
    path = "model/checkpoints/%s_%s.pt" % (
        model.__class__.__name__, time.strftime("%Y%m%d%H%M%S", train_time)
    )
    path = ROOT_DIR.joinpath(path)
    torch.save(model, path)
    logger.info(f"model saved: {path}")


def load_model(path: str):
    path = ROOT_DIR.joinpath(path)
    # load model to cpu as default.
    model = torch.load(path, map_location=torch.device('cpu'))
    return model


def load_reviews2(review: Dict[str, DataFrame], query_id: str, exclude_id: str, max_length) -> List[int]:
    """
    1. Load review from review dict by userID/itemID
    2. Exclude unknown review by itemID/userID..
    3. Pad review text to max_length

    E.g. get all reviews written by user1 except itemA
         when we predict the rating of itemA marked by user1.

        DataFrame for user1:

            | itemID | review |
            | itemA  | 0,1,2  |
            | itemB  | 1,2,3  |
            | itemC  | 2,3,4  |

        query_id: user1
        exclude_id: itemA
        max_length: 8

        output = [1, 2, 3, 2, 3, 4, PAD_WORD_ID, PAD_WORD_ID]
    """

    # "posID", "posReview" USER
    # "userID", "posReview" POSITEM
    # "userID", "negReview" NEGITEM

    reviews = review[query_id]

    if ("userID" in reviews.columns) and ("posReview" in reviews.columns):
        key = "userID"
        reviews = reviews["posReview"][reviews[key] != exclude_id].to_list()
    elif "posID" in reviews.columns:
        key = "posID"
        reviews = reviews["posReview"][reviews[key] != exclude_id].to_list()
    else:
        key = "userID"
        reviews = reviews["negReview"][reviews[key] != exclude_id].to_list()

    reviews = list(chain.from_iterable(reviews))

    if len(reviews) >= max_length:
        reviews = reviews[:max_length]
    else:
        reviews = reviews + [PAD_WORD_ID] * (max_length - len(reviews))
    return reviews


def get_data_loader2(data: DataFrame, config: BaseConfig):
    logger.info("Generating data iter...")
    _, _, review_by_user2, review_by_positem2, review_by_negitem2 = main_preprocessing()

    user_reviews = [torch.LongTensor(load_reviews2(review_by_user2, userID, posID, config.max_review_length))
                    for userID, posID in zip(data["userID"], data["posID"])]
    user_reviews = torch.stack(user_reviews)

    review_by_positem2 = [torch.LongTensor(load_reviews2(review_by_positem2, posID, userID, config.max_review_length))
                          for userID, posID in zip(data["userID"], data["posID"])]
    review_by_positem2 = torch.stack(review_by_positem2)

    review_by_negitem2 = [torch.LongTensor(load_reviews2(review_by_negitem2, negID, userID, config.max_review_length))
                          for userID, negID in zip(data["userID"], data["negID"])]
    review_by_negitem2 = torch.stack(review_by_negitem2)

    # ratings = torch.Tensor(data["rating"].to_list()).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(user_reviews, review_by_positem2, review_by_negitem2)
    pin_memory = config.device not in ["cpu", "CPU"]
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=pin_memory)
    logger.info("Data iter loaded.")
    return data_iter


def get_data_loader_test(data: DataFrame, review_by_user, review_by_item, config: BaseConfig):
    logger.info("Generating data iter...")

    user_reviews = [torch.LongTensor(th.load_reviews(review_by_user, userID, itemID, config.max_review_length))
                    for userID, itemID in zip(data["userID"], data["itemID"])]
    user_reviews = torch.stack(user_reviews)

    item_reviews = [torch.LongTensor(th.load_reviews(review_by_item, itemID, userID, config.max_review_length))
                    for userID, itemID in zip(data["userID"], data["itemID"])]
    item_reviews = torch.stack(item_reviews)

    ratings = torch.Tensor(data["rating"].to_list()).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(user_reviews, item_reviews, ratings)
    pin_memory = config.device not in ["cpu", "CPU"]
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=pin_memory)
    logger.info("Data iter loaded.")
    return data_iter


def eval_model2(model: BaseModel, data_iter: DataLoader) -> float:
    model.eval()
    model_name = model.__class__.__name__
    config: BaseConfig = model.config
    logger.debug("Evaluating %s..." % model_name)
    with torch.no_grad():
        pospredicts = []
        negpredicts = []
        for batch_id, iter_i in enumerate(data_iter):
            user_review, positem_review, negitem_review = iter_i
            user_review = user_review.to(config.device)
            positem_review = positem_review.to(config.device)
            negitem_review = negitem_review.to(config.device)
            pos_predict = model(user_review, positem_review)
            neg_predict = model(user_review, negitem_review)
            pospredicts.append(pos_predict)
            negpredicts.append(neg_predict)

        pospredicts = torch.cat(pospredicts)
        negpredicts = torch.cat(negpredicts)
        loss = -torch.mean(torch.nn.functional.logsigmoid(pospredicts - negpredicts))
        return loss.item()


def eval_model_test(model: BaseModel, data_iter: DataLoader) -> float:
    model.eval()
    model_name = model.__class__.__name__
    config: BaseConfig = model.config
    logger.debug("Evaluating %s..." % model_name)
    with torch.no_grad():
        predicts = []
        ratings = []
        for batch_id, iter_i in enumerate(data_iter):
            user_review, item_review, _ = iter_i
            user_review = user_review.to(config.device)
            item_review = item_review.to(config.device)
            predict = model(user_review, item_review)
            predicts.append(predict)

        predicts = torch.cat(predicts)
        list_predicts = predicts.tolist()
        return list_predicts


def train_model3(model: BaseModel, train_data: DataFrame):
    model_name = model.__class__.__name__
    train_time = time.localtime()
    add_log_file(logger, "log/%s_%s.log" % (model_name, time.strftime("%Y%m%d%H%M%S", train_time)))
    logger.info("Training %s..." % model_name)

    config: BaseConfig = model.config
    logger.info(config.__dict__)
    model.to(config.device)

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_s = lr_scheduler.ExponentialLR(opt, gamma=config.learning_rate_decay)
    loss = torch.nn.MSELoss()

    last_progress = 0.
    last_loss = float("inf")
    train_data_iter2 = get_data_loader2(train_data, config)
    # dev_data_iter = get_data_loader(dev_data, config)
    batches_num = math.ceil(len(train_data) / float(config.batch_size))

    while model.current_epoch < config.num_epochs:

        model.train()

        for batch_id, iter_i in enumerate(train_data_iter2):
            user_review, positem_review, negitem_review = iter_i
            user_review = user_review.to(config.device)
            positem_review = positem_review.to(config.device)
            negitem_review = negitem_review.to(config.device)
            opt.zero_grad()
            pos_predict = model(user_review, positem_review)
            neg_predict = model(user_review, negitem_review)
            distances = torch.abs(pos_predict - neg_predict)
            # li = - torch.sum(torch.log(torch.sigmoid(distances)))
            li = -torch.mean(torch.nn.functional.logsigmoid(pos_predict - neg_predict))
            li.backward()
            opt.step()

            # log progress
            current_batches = model.current_epoch * batches_num + (batch_id + 1.0)
            total_batches = config.num_epochs * batches_num
            progress = current_batches / total_batches
            if progress - last_progress > 0.001:
                logger.debug("epoch %d, batch %d, loss: %f (%.2f%%)" %
                             (model.current_epoch, batch_id, li.item(), 100.0 * progress))
                last_progress = progress

        # complete one epoch
        train_loss = eval_model2(model, train_data_iter2)
        # dev_loss = eval_model(model, dev_data_iter, loss)
        logger.info("Epoch %d complete. Total loss(train)=%f" % (model.current_epoch, train_loss))

        # save best model
        if train_loss < last_loss:
            last_loss = train_loss
            save_model(model, train_time)

        lr_s.step(model.current_epoch)
        model.current_epoch += 1

    logger.info("%s trained!" % model_name)
    remove_log_file(logger)


config = DeepCoNNConfig(
    num_epochs=2,
    batch_size=2,
    learning_rate=1e-4,
    l2_regularization=1e-3,
    learning_rate_decay=0.95,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    max_review_length=2048,  # Make sure this value is smaller than max_length in data_reader.py
    word_dim=WORD_EMBEDDING_SIZE,
    kernel_widths=[2, 3, 5, 7],
    kernel_deep=100,
    latent_factors=50,
    fm_k=8
)


def final_test(config1):
    _, test, _, _, _ = main_preprocessing()
    word_vec = dr.get_word_vec()
    review_by_user_test, review_by_item_test = dr.get_reviews_in_idx(test, word_vec)
    test_data_iter = get_data_loader_test(test, review_by_user_test, review_by_item_test, config1)
    model = load_model("model/checkpoints/DeepCoNN_20221026204239.pt")
    list_predicts = eval_model_test(model, test_data_iter)
    final_test_df = pd.DataFrame()
    final_test_df["userID"] = test["userID"]
    final_test_df["itemID"] = test["itemID"]
    final_test_df["predicted_ratings"] = list_predicts
    return final_test_df


# per training con bpr
#new_train, test, review_by_user2, review_by_positem2, review_by_negitem2 = main_preprocessing()
#model = DeepCoNN(config, load_embedding_weights())
#train_model3(model, new_train)

# per test csv
final_df = final_test(config)
final_df.to_csv("final_dataframe1e-3.csv")
