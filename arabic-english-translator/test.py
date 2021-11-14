import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, TimeDistributed, Embedding, LSTM, RepeatVector
from tensorflow.keras.models import load_model, Sequential 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import  pad_sequences
from tensorflow.keras.utils import  to_categorical
from pickle import  dump, load
import json
import re
import os 

def load_doc(filename):
    doc = open(filename, mode='r', encoding='utf-8') 
    text = doc.read()
    return text
def ara_eng(text):
    """
    Take the raw text and remove un used text and return 2D array each array will have [ara, eng]
    """
    lines = text.split('\n')
    pairs = []
    for line in lines:
        line = line.split('\t')
        if len(line) >= 3:
            eng, ara = line[0], line[1]
            pairs.append(np.array([ara, eng]))
    return np.array(pairs)

def clean_text(pairs):
    """
    1. Make all English letter lowercase
    2. Remove marks like !,?,.
    3. Remove numbers
    4. Convert 'm to " am" 
    5. Convert 've to " have"
    6. Convert 're to " are"
    7. Convert n't to ' not'
    8. Convert 's to " is"
    9. Convert 'd to " would"
    10. Convert 'll to " will" 
    """
    cleaned_pairs = []
    for pair in pairs:
        ara, eng = pair[0], pair[1]
        #  Make all English letter lowercase
        eng = eng.lower()
        # Remove marks like !,?,.
        eng = re.sub(r"!|\?|\.", '', eng)
        # Remove ! from Arabic words 
        ara = re.sub(r"!", '', ara)
        # Remove shortcuts 
        eng = re.sub(r"'m", ' am', eng)
        eng = re.sub(r"'ve", ' have', eng)
        eng = re.sub(r"'re", ' are', eng)
        eng = re.sub(r"'n't", ' not', eng)
        eng = re.sub(r"'s", ' is', eng)
        eng = re.sub(r"'d", ' would', eng)
        eng = re.sub(r"'ll", ' will', eng)
        cleaned_pairs.append(np.array([ara, eng]))
    return np.array(cleaned_pairs)
    
def save_clean_data(data, filename):
    dump(data, open(filename, 'wb'))

def load_clean_data(filename):
    return load(open(filename, 'rb'))

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_len(lines):
    return max([len(line.split()) for line in lines])

def encode_sequences(tokenizer, length, lines):
    sequences = tokenizer.texts_to_sequences(lines)
    sequences = pad_sequences(sequences=sequences, maxlen=length, padding='post')
    return sequences

# One hot encode for the output
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    return np.array(ylist)

def word_for_id(integer, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == integer:
            return word
    return None

filename = 'data.txt'
doc = load_doc(filename)
pairs = ara_eng(doc)
cleaned_pairs = clean_text(pairs)

print("Sample of the cleaned data")
for i in range(20):
    print("%s => %s " % (cleaned_pairs[i][0], cleaned_pairs[i][1]))

raw_dataset = cleaned_pairs

n_sentences = 12000
dataset = raw_dataset[:n_sentences, :]
print(dataset.shape)
# random shuffle
np.random.shuffle(dataset)
# Split data 
train, test = dataset[:10000], dataset[10000:]

print(train.shape, test.shape)

eng = dataset[:, 1]
eng_tokenizer = create_tokenizer(eng)
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_len(eng)
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))

ara = dataset[:, 0]
ara_tokenizer = create_tokenizer(ara)
ara_vocab_size = len(ara_tokenizer.word_index) + 1
ara_length = max_len(ara)
print('Arabic Vocabulary Size: %d' % ara_vocab_size)
print('Arabic Max Larath: %d' % (ara_length))

# prepare training data
trainX = encode_sequences(ara_tokenizer, ara_length, train[:, 0])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 1])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(ara_tokenizer, ara_length, test[:, 0])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 1])
testY = encode_output(testY, eng_vocab_size)

model = load_model('model.h5')
preds = model.predict(testX)
print(preds.shape)

save_clean_data(preds, 'preds.pkl')
print('Preds Saved /download/arabic-english-translator/preds.pkl')