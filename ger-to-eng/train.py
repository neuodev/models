import os
import string 
import re 
from pickle import dump 
from unicodedata import normalize 
from numpy import array
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt 
import click

# Datasets Source  http://www.manythings.org/anki/

# Load doc into memory
def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

# Split a loaded document into sentences 
def to_paris(doc):
    lines = doc.split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs 

# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)

def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print("Saved: %s" % filename)

filename = 'data.txt'
doc = load_doc(filename)
pairs = to_paris(doc)
cleaned_pairs = clean_pairs(pairs)
print("All Pairs: ", cleaned_pairs.shape)

# Select a small sample for now 
n_sentences = 10000 
dataset = cleaned_pairs[:n_sentences, :]
# Shuffle the data
np.random.shuffle(dataset)

# Split data into test and train 
train, test = dataset[:9000], dataset[9000:]
print("Train: ", train.shape)
print("Test: ", test.shape)

# Fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max([len(line.split()) for line in lines])

def encode_sequences(tokenizer, length, lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

# one hot encode target sequence 
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model

# Prepare english tokenizer 
eng_idx = 0 
ger_idx = 1
eng_tokenizer = create_tokenizer(dataset[:, eng_idx])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, eng_idx])

print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))

# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)

def train_model(epochs):
    model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print(model.summary())
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=64, validation_data=(testX, testY), verbose=1).history
    print('Save the model\n')
    model.save('model.h5')
    return history

def predict_sequence(model, tokenizer, source):
    preds = model.predict(source, verbose=1)
    print('Preds: ', preds.shape)
    integers = [np.argmax(vector) for vector in preds]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

def evaludate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        source = source.reshape(1, source.shape[0])
        print("source: ", source)
        translation = predict_sequence(model, tokenizer, sources)
        print("Translation: ", translation)
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append([raw_target.split()])
        predicted.append(translation.split())
    for i in range(len(actual)):
        print("Actual:\n", actual[i])
        print("Predicted: \n", predicted[i])



@click.command(name='train')
@click.option('--mode', default='train', help='Traning or testing')
@click.option('--epochs', default=10, help='Number of epochs')
@click.option('--plot', default=False, help='Plot model traning')
def train_model(mode, epochs, plot):

    if mode == 'train':
        history = train_model(epochs)
        # Plot traning history
        if plot:
            plt.plot(history['loss'], label='Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.show()
    elif mode == 'test':
        model = load_model('model.h5')
        print('Evaluate on the traing data\n')
        evaludate_model(model, eng_tokenizer, trainX, train)
        print('Evaluate on the testing data\n')
        evaludate_model(model, eng_tokenizer, testX, test)

if __name__ == '__main__':
    train_model()
