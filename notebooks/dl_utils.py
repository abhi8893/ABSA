import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import pickle
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import spacy



def prepare_dl_input_file(raw_file, out_file):
    pass

def read_dl_input_file(domain, subset, year='2014'):
        
    fname =f"../data/processed/SemEval{year}/{domain}_{subset}_dl.txt"
    with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        lines = f.readlines()
        
    return lines



def create_word_vec(word2idx, embed_dim):
    
    emb_file = f'../data/embeddings/glove.6B.{embed_dim}d.txt'

    n = len(word2idx)
    w2v = {}
    
    with open(emb_file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word2idx:
                w2v[word] = np.asarray(values[1:], dtype='float32')
                                
    return w2v


import nltk

import nltk

UNIVERSAL_TAGS = {
    "VERB": 1,
    "NOUN": 2,
    "PRON": 3,
    "ADJ": 4,
    "ADV": 5,
    "ADP": 6,
    "CONJ": 7,
    "DET": 8,
    "NUM": 9,
    "PRT": 10,
    "X": 11,
    ".": 12,
}

MODIFIED_TAGS = {
    "VERB": 1,
    "NOUN": 2,
    "ADJ": 3,
    "ADV": 4,
    "CONJ": 5,
    "DET": 6,
}

import numpy as np

def pos_tag(word):
    return nltk.pos_tag([word], tagset='universal')[0][1]

def get_onehot_pos(word):
    
    tag = pos_tag(word)
    arr = np.zeros(6)
    idx = MODIFIED_TAGS.get(tag, None)
    if idx is not None:
        arr[idx-1] = 1
        
    return arr


import gensim 

def create_embedding_matrix(word2idx, embed_dim=None, embed_type='glove', concat_pos_tag=False):
    
    if embed_type == 'glove.wiki':
        emb_file = f'../data/embeddings/glove.wiki/glove.6B.{embed_dim}d.txt'
        
    elif embed_type == 'glove.twitter':
        emb_file = f'../data/embeddings/glove.twitter/glove.twitter.27B.{embed_dim}d.txt'

        
    elif embed_type == 'amazon':
        emb_file = '../data/embeddings/AmazonWE/sentic2vec.txt'
        embed_dim = 300
        
    elif embed_type == 'google':
        emb_file = '../data/embeddings/GoogleNews-vectors-negative300.txt'
        embed_dim = 300
        
    elif embed_type in ['restaurants', 'laptops']:
        emb_file = f'../data/embeddings/domain_embedding/{embed_type}_emb.vec.bin'
        embed_dim = 100
        
        

    n = len(word2idx)
    if concat_pos_tag:
        matrix_dim = embed_dim + 6
    else:
        matrix_dim = embed_dim
    
    embedding_matrix = np.zeros((n + 1, matrix_dim))
    
    
    i = 0
    
    if embed_type in ['restaurants', 'laptops']:
        domain_model = gensim.models.fasttext.load_facebook_model(emb_file)
        
        for word in word2idx:
            
            try:
                idx = word2idx[word]
                embedding_matrix[idx][:embed_dim] = domain_model[word]
                if concat_pos_tag:
                    embedding_matrix[idx][embed_dim:] = get_onehot_pos(word)
                i += 1
                    
            except Exception:
                print(idx, word)
            
            
                
    else:       
        with open(emb_file, 'r') as f:
            for line in f:

                values = line.split()
                word = values[0]

                if word in word2idx:
                    idx = word2idx[word]
                    embedding_matrix[idx][:embed_dim] = np.asarray(values[1:], dtype='float32')
                    if concat_pos_tag:
                        embedding_matrix[idx][embed_dim:] = get_onehot_pos(word)

                    i += 1
                
    pct_vocab = i*100/n
                
    print(f'Word vectors found for {pct_vocab:.2f}% of vocabulary')
                                
    return embedding_matrix



import re
import string

def remove_punct(s):
    return s.translate(str.maketrans('', '', string.punctuation))

def decontracted(phrase):

    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase




def split_reviews_by_aspects(dl_input_lines: list):
    
    reviews_raw = []
    reviews_raw_without_aspects = []
    reviews_left = []
    reviews_left_with_aspects = []
    reviews_right = []
    reviews_right_with_aspects = []
    postags_raw = []
    postags_raw_left_with_aspects = []
    postags_raw_right_with_aspects = []
    aspects = []
    polarities = []
    
    nlp = spacy.load("en_core_web_sm")
    
    for i in range(0, len(dl_input_lines), 3):
        review = decontracted(dl_input_lines[i])
        review_left, _, review_right = [s.lower().strip() for s in review.partition("$T$")]
        aspect = dl_input_lines[i+1].lower().strip()
        polarity = dl_input_lines[i+2].strip()

        review_raw = ' '.join([review_left, aspect, review_right])
        review_raw = re.sub(' +', ' ', review_raw)
        
 
        
        reviews_raw.append(review_raw)
        reviews_raw_without_aspects.append(review_left + " " + review_right)
        reviews_left.append(review_left)
        reviews_left_with_aspects.append(review_left + " " + aspect)
        reviews_right.append(review_right)
        reviews_right_with_aspects.append(aspect + " " + review_right)
        aspects.append(aspect)
        polarities.append(int(polarity))
        
        
    res = {
        'reviews_raw': reviews_raw,
        'reviews_raw_without_aspects': reviews_raw_without_aspects,
        'reviews_left': reviews_left,
        'reviews_left_with_aspects': reviews_left_with_aspects,
        'reviews_right': reviews_right,
        'reviews_right_with_aspects': reviews_right_with_aspects,
        'aspects': aspects,
        'polarities': polarities,
        'postags_raw': postags_raw
    }
        
        
    return res

def create_sequence_data(texts, maxlen, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen, padding='post', truncating='post')


def prepare_data_for_dl(domain='restaurants', subset='train', embed_dim=300, embed_type='glove', 
                        max_input_len=70, max_aspect_len=5, num_classes=3, tokenizer=None, concat_pos_tag=False):
    
    
    if domain == 'both':
        lines_rest = read_dl_input_file('restaurants', subset)
        lines_lap = read_dl_input_file('laptops', subset)
        lines = lines_rest + lines_lap
        
    else:
        # Read the lines from the pre-formatted dl input file
        if subset == 'train':
            lines = read_dl_input_file(domain, subset) #+ read_dl_input_file(domain, 'test')
        else:
            lines = read_dl_input_file(domain, 'test')
    
    # now obtain the splitted reviews on the left and right side of the aspect
    spltd = split_reviews_by_aspects(lines)
    polarities = spltd.pop('polarities')
    postags_raw = spltd.pop('postags_raw')
    
    # Tokenize 
    if subset == 'test':
        if tokenizer is None:
            raise ValueError('Provide a tokenizer fitted on the train data!')
        if max_input_len is None:
            raise ValueError('Provide a maximum input length for padding the input sequence!')
        if max_aspect_len is None:
            raise ValueError('Provide a maximum aspect length for padding the aspect terms!')
            
    elif subset == 'train':
        tokenizer = Tokenizer(lower=False)
        tokenizer.fit_on_texts(spltd['reviews_raw'])
        
        
    word2idx = tokenizer.word_index
    
    # Create sequence padded data of indices
    res = {}
    
    for k, v in spltd.items():
        if k == 'aspects':
            maxlen = max_aspect_len
        else:
            maxlen = max_input_len
            
        res[f'{k}_idx'] = create_sequence_data(v, maxlen, tokenizer)
        
    # one hot encode polarities
    res['polarity_ohe'] = to_categorical(polarities, num_classes)
    res['postags_raw'] = postags_raw
        
    if subset == 'test':
        return res
    
    
    
    if type(embed_type) is list:
        embedding_matrix = []
        for emb_type in embed_type:
            embedding_matrix.append(create_embedding_matrix(word2idx, embed_dim, emb_type, concat_pos_tag))
            concat_pos_tag = False
            
        embedding_matrix = np.hstack(embedding_matrix)
        
    else:
        embedding_matrix = create_embedding_matrix(word2idx, embed_dim, embed_type, concat_pos_tag)
                                  
    res['embedding_matrix'] = embedding_matrix
    res['tokenizer'] = tokenizer


    return res

