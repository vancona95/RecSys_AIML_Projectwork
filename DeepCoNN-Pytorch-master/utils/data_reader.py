import pickle
from typing import Set, List, Dict
import sys
import nltk
nltk.download('wordnet')
import pandas
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from log_hepler import logger
from path_helper import ROOT_DIR
from word2vec_hepler import review2wid, PAD_WORD, get_word_vec, save_embedding_weights


def get_all_data(path="data/reviews.json") -> DataFrame:
    return pandas.read_json(ROOT_DIR.joinpath(path), lines=True)


def get_train_dev_test_data(path="data/reviews.json") -> (DataFrame, DataFrame, DataFrame):
    all_data = get_all_data(path)
    train, test = train_test_split(all_data, test_size=0.2, random_state=42)
    train, dev = train_test_split(train, test_size=0.1, random_state=42)
    return train, dev, test


def get_stop_words(path="data/stopwords.txt") -> Set[str]:
    with open(ROOT_DIR.joinpath(path)) as f:
        return set(f.read().splitlines())


def get_punctuations(path="data/punctuations.txt") -> Set[str]:
    with open(ROOT_DIR.joinpath(path)) as f:
        return set(f.read().splitlines())


def process_raw_data(in_path="data/AMAZON_FASHION_5.json", out_path="data/reviews.json"):
    """
    Read raw data and remove useless columns and clear review text.
    Then save the result to file system.
    """

    logger.info("reading raw data...")
    df = pandas.read_json(ROOT_DIR.joinpath(in_path), lines=True)
    df = df[["reviewerID", "asin", "reviewText", "overall"]]
    df.columns = ["userID", "itemID", "review", "rating"]
    stop_words = get_stop_words()
    punctuations = get_punctuations()
    lemmatizer = nltk.WordNetLemmatizer()

    def clean_review(review: str):
        review = str(review)
        review = review.lower()
        assert "'" not in punctuations
        for p in punctuations:
            review = review.replace(p, " ")
        tokens = review.split()
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    logger.info("cleaning review text...")
    df["review"] = df["review"].apply(clean_review)
    df.to_json(ROOT_DIR.joinpath(out_path), orient="records", lines=True)
    logger.info("Processed data saved.")


def get_reviews_in_idx(data: DataFrame, word_vec) -> (Dict[str, DataFrame], Dict[str, DataFrame]):
    """
    1. Group review by user and item.
    2. Convert word into word idx.
    :return The dictionary from userID/itemID to review text in word idx with itemID/userID.
    """

    data["review"] = data["review"].apply(review2wid, args=[word_vec])
    review_by_user = dict(list(data[["itemID", "review"]].groupby(data["userID"])))
    review_by_item = dict(list(data[["userID", "review"]].groupby(data["itemID"])))

    return review_by_user, review_by_item


def get_review_dict():
    user_review = pickle.load(open(ROOT_DIR.joinpath("data/user_review_word_idx.p"), "rb"))
    item_review = pickle.load(open(ROOT_DIR.joinpath("data/item_review_word_idx.p"), "rb"))
    return user_review, item_review


if __name__ == "__main__":
    process_raw_data()

    train_data, dev_data, test_data = get_train_dev_test_data()
    known_data = pandas.concat([train_data, dev_data])

    word_vec = get_word_vec()
    save_embedding_weights(word_vec)

    user_review, item_review = get_reviews_in_idx(known_data, word_vec)
    pickle.dump(user_review, open(ROOT_DIR.joinpath("data/user_review_word_idx.p"), "wb"))
    pickle.dump(item_review, open(ROOT_DIR.joinpath("data/item_review_word_idx.p"), "wb"))
