from typing import Set, List

import numpy as np
import pandas
import torch
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors

from log_hepler import logger
from path_helper import ROOT_DIR

PAD_WORD = "<pad>"
PAD_WORD_ID = 3000000
WORD_EMBEDDING_SIZE = 300


def review2wid(review: str, word_vec: Word2VecKeyedVectors) -> List[int]:
    """
    Convert words in review to word idx.
    The idx is from pre-trained word embedding model.
    """

    wids = []
    for word in review.split():
        if word in word_vec:
            wid = word_vec.key_to_index[word]
        else:
            wid = word_vec.key_to_index[PAD_WORD]
        wids.append(wid)
    return wids


def get_word_vec(path='data/GoogleNews-vectors-negative300.bin'):
    """
    Read pre-trained word embedding model, and add "<pad>" to it with zero weight.
    """

    logger.info("loading word2vec model...")
    path = ROOT_DIR.joinpath(path)
    word_vec = KeyedVectors.load_word2vec_format(path, binary=True)
    word_vec.add_vectors([PAD_WORD], np.zeros([1, 300]))
    logger.critical(f"PAD_WORD_ID is {word_vec.key_to_index[PAD_WORD]}.")
    logger.info("word2vec model loaded.")
    return word_vec


def save_embedding_weights(word_vec, out_path="data/embedding_weight.pt"):
    """
    Save the weights of pre-trained word embedding model to file.
    Thus we don't need to load it when train our model.
    This helps to save RAM and model init time.
    """

    weight = torch.Tensor(word_vec.vectors)
    torch.save(weight, ROOT_DIR.joinpath(out_path))
    logger.info("Word embedding weight saved.")


def load_embedding_weights(path="../data/embedding_weight.pt"):
    return torch.load(path)


# Find the unknowns words in review text.
# This step is not necessary for model train.
if __name__ == "__main__":
    df = pandas.read_json(ROOT_DIR.joinpath("data/reviews.json"), lines=True)
    word_vec = get_word_vec()
    unknown_words: Set[str] = set()
    for review in df["review"]:
        for word in review.split():
            if word not in word_vec:
                unknown_words.add(word)

    logger.warning(f"{len(unknown_words)} unknown words!")
    with open(ROOT_DIR.joinpath("out/UNKs.txt"), "w", encoding="utf-8") as f:
        for word in unknown_words:
            f.write(f"{word}\n")
