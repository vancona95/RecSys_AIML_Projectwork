# RecSys_AIML_Projectwork

_Credits: Vincenzo Ancona, Giandomenico Misciagna and Giuseppe Colacicco_
# DeepCoNN_BPR
This is a BPR adaptation of the DeepCoNN PyTorch implementation.

The Github repository of the original DeepCoNN PyTorch implementation can be found from:

https://github.com/KindRoach/DeepCoNN-Pytorch

Paper of the original model:

_Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation. In WSDM. ACM, 425-434._

All the Python modules (both from the original PyTorch implementation and DeepCoNN_BPR) can be found in the `utils` folder.
## Before Running Code

#### Get Data
Download and unzip "Amazon Fashion" data set from :  
https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz 

Then put it under the path `data`

#### Get Pre-trained Word Embedding Model
We use GoogleNews-vectors-negative300.bin as pre-trained word embedding model.
You could find it at:  
https://code.google.com/archive/p/word2vec/  
Then put it under the path `data`

#### Environments
```
pandas~=1.0.3
numpy~=1.18.1
gensim~=3.8.0
pytorch~=1.3.1
nltk~=3.4.5
scikit-learn~=0.22.1
```

## Train & Test of DeepCoNN_BPR

In order to run our DeepCoNN_BPR implementation, simply run the "DeepCoNN_BPR.py" module:
```
python DeepCoNN_BPR.py
```

#### Train Model
In order to train the model, uncomment the following code lines in the main function:

```

new_train, test, review_by_user2, review_by_positem2, review_by_negitem2 = main_preprocessing()
    
model = DeepCoNN(config, load_embedding_weights())
    
train_model_for_BPR(model, new_train)

```
You will find trained model file in `model/checkpoints`

#### Test Model
In order to test the model, uncomment the following code lines in the main function:

```

modelpath = "model/checkpoints/DeepCoNN_20221027183344.pt"
    
final_df = final_test(config, modelpath)
    
final_df.to_csv("final_dataframe1e-5.csv")

```

Replace the model path in the first line and the path of the output csv file in the last line.

###

## Train & Test of the original implementation

In order to run the original DeepCoNN PyTorch implementation, follow the instructions below:

#### Data Pre-processing
```
python data_reader.py
```

#### Train Model
```
python train.py
```
You will find trained model file in `model/checkpoints`

#### Eval Model  
Replace the model path in `eval.py` at first.
```
python eval.py
