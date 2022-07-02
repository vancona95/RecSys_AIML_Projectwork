import json
import pandas as pd
import numpy
import re

file = pd.read_json("AMAZON_FASHION_5.json", lines = True)

df = pd.DataFrame(file)
selected_columns = df[["reviewerID", "asin", "overall"]]
raw_rating_data = selected_columns.copy()
raw_rating_data2 = raw_rating_data.rename(columns = {'reviewerID': 'user id', 'asin': 'item id', 'overall': 'rating'})
raw_rating_data2.to_csv("raw_rating_data.csv", index=None, header=None, sep=':')

with open("raw_rating_data.csv", "r") as sources:
    lines = sources.readlines()
with open("raw_rating_data.csv", "w") as sources:
    for line in lines:
        r = re.compile(":")
        sources.write(r.sub('::', line))

selected_columns = df[["asin", "reviewText"]]
raw_item_document = selected_columns.copy()
raw_item_document2 = raw_item_document.rename(columns = {'asin': 'item id', 'reviewText': 'text'})
raw_item_document2 = raw_item_document2.groupby(['item id'])['text'].apply(list).reset_index()
raw_item_document3= pd.DataFrame(raw_item_document2.text.values.tolist()).add_prefix('text')
itemids = raw_item_document2['item id']
raw_item_document3.insert(0, 'item id', itemids)
raw_item_document3.to_csv("raw_item_document.csv", index=None, header =None, sep = '|')

with open("raw_item_document.csv", "r") as sources:
    lines = sources.readlines()
with open("raw_item_document.csv", "w") as sources:
    for line in lines:
        r = re.compile("\|")
        sources.write(r.sub('::', line, 1))




