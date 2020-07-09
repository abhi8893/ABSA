import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import pickle
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical



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


def create_embedding_matrix(word2idx, embed_dim):
    
    emb_file = f'../data/embeddings/glove.6B.{embed_dim}d.txt'

    n = len(word2idx)
    embedding_matrix = np.zeros((n + 1, embed_dim))
    
    with open(emb_file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word2idx:
                idx = word2idx[word]
                embedding_matrix[idx] = np.asarray(values[1:], dtype='float32')
                                
    return embedding_matrix



def split_reviews_by_aspects(dl_input_lines: list):
    
    reviews_raw = []
    reviews_raw_without_aspects = []
    reviews_left = []
    reviews_left_with_aspects = []
    reviews_right = []
    reviews_right_with_aspects = []
    aspects = []
    polarities = []

    for i in range(0, len(dl_input_lines), 3):
        review_left, _, review_right = [s.lower().strip() for s in dl_input_lines[i].partition("$T$")]
        aspect = dl_input_lines[i+1].lower().strip()
        polarity = dl_input_lines[i+2].strip()

        review_raw = ' '.join([review_left, aspect, review_right])

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
        'polarities': polarities
    }
        
        
    return res


def create_sequence_data(texts, maxlen, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen, padding='post', truncating='post')



def prepare_data_for_dl(domain='restaurants', subset='train', embed_dim=50, 
                        max_input_len=70, max_aspect_len=5, num_classes=3, tokenizer=None):
    
    
    # Read the lines from the pre-formatted dl input file
    lines = read_dl_input_file(domain, subset)
    
    # now obtain the splitted reviews on the left and right side of the aspect
    spltd = split_reviews_by_aspects(lines)
    polarities = spltd.pop('polarities')
    
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
        
    if subset == 'test':
        return res

    res['embedding_matrix'] = create_embedding_matrix(word2idx, embed_dim)
    res['tokenizer'] = tokenizer


    return res

