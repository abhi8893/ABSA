import pandas as pd
from spacy.tokenizer import Tokenizer
from spacy import load
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import re
from collections import Counter
import pandas as pd
import keras.backend as K


RAW_DATA_FILE_2014_TRAIN = "../data/raw/SemEval2014/{domain}_train_v2.csv"
RAW_DATA_FILE_2014_TEST = "../data/raw/SemEval2014/{domain}_test.csv"

RAW_DATA_FILE_2016_TRAIN = "../data/raw/SemEval2016/ABSA16_{domain}_train_SB1.csv"
RAW_DATA_FILE_2016_TEST = "../data/raw/SemEval2016/EN_{domain}_SB1_test_gold.csv"


RAW_DATA_FILE = {
    '2014': {'train': RAW_DATA_FILE_2014_TRAIN, 'test': RAW_DATA_FILE_2014_TEST},
    '2016': {'train': RAW_DATA_FILE_2016_TRAIN, 'test': RAW_DATA_FILE_2016_TEST}
}

def load_raw_file(domain, subset='train', year='2014'):
    f = RAW_DATA_FILE[year][subset].format(domain=domain)    
 
    return pd.read_csv(f)


nlp = load("en")
tokenizer = Tokenizer(nlp.vocab)


# Remove Stopwords
# Custom stopword list?
def remove_stopwords(sentence) :
    return " ".join([str(token) for token in tokenizer(sentence.replace(',', '').replace(".","").lower())
                     if not token.is_stop and not token.is_punct and not token.is_digit and token.is_alpha])


def lemmatize(sentence):
    return " ".join([token.lemma_ for token in nlp(sentencence)])

def remove_aspect(text, aspect) :
    pattern = '\s*'+aspect.replace('(', '\(').replace(')', '\)')+'\s*'
    return re.sub(pattern, ' ', text)


def split_text(text, on, method):
    spltd = text.split(on)
    
    if method == 'before':
        res = spltd[0]
    elif method == 'after':
        if len(spltd) > 1: 
            res = spltd[1]
        else:
            res = ' '
    
    return res

# split and get left side of the sentence
def split_left(text_splitpoint) :
    sentence, split_point = text_splitpoint
    return sentence.split(split_point)[0]


# split and get right side of the sentence
def split_right(text_splitpoint):
    sentence, split_point = text_splitpoint
    split = sentence.split(split_point)
    return split[1] if len(split)>1 else " "


def remove_polarity(polarity, df):
    return df.loc[df.polarity != polarity, :]


def preprocess(df):
    
    df = df.copy()
    
    cols = ['text', 'term', 'polarity']
    df = remove_polarity('conflict', df).loc[:, cols]
    
   # remove stopwords
    df.loc[:, 'text'] = df.text.apply(remove_stopwords)

    # lemmatize
    # texts = []
    # for doc in nlp.pipe(df.text):
    #     sent_lemmatized = ' '.join([token.lemma_ for token in doc])
    #     texts.append(sent_lemmatized)

    # df.loc[:, 'text'] = texts

    # extract before aspect and after aspect sentence
    df['before_aspect'] = df.loc[:, ['text', 'term']].apply(
                          lambda r: split_text(r['text'], r['term'], 'before'), axis=1)

    df['after_aspect'] = df.loc[:, ['text', 'term']].apply(
                          lambda r: split_text(r['text'], r['term'], 'after'), axis=1)

    # remove aspect 
    df['without_aspect'] = df.loc[:, ['text', 'term']].apply(
                          lambda r: remove_aspect(r['text'], r['term']), axis=1)
    
    
    return df


def f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

