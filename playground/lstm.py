
from audioop import bias
from string import whitespace
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding,SpatialDropout1D
from keras.initializers import Constant


vocab_size = 10000
embedding_dim = 50
max_length = 15
training_size = 25000


embedding_dict = {}
with open('glove/glove.6B.50d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()



columnNames = ["headline", "clickbait"]
# df = pd.read_csv(
#     'datasets/clickbait_data.csv')

# clickbait_dataset = df.sample(frac=1)
# clickbait_dataset.to_csv("datasets/randomized_dataset.csv")
clickbait_dataset = pd.read_csv('datasets/clickbait_data.csv')

inputTitles = clickbait_dataset.headline.to_list()
clickbait = clickbait_dataset.clickbait.to_list()


training_sentences = inputTitles[0:training_size]
testing_sentences = inputTitles[training_size:30000]
training_clickbait = clickbait[0:training_size]
testing_clickbait = clickbait[training_size:30000]



tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
numUniqueWords = len(word_index)+1

embedding_matrix = np.zeros((numUniqueWords,50))
for word, i in word_index.items():
    if i>numUniqueWords:
        continue
    emb_vector = embedding_dict.get(word)
    if emb_vector is not None:
        embedding_matrix[i] = emb_vector


training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences, maxlen=max_length, padding='post', truncating='post')


testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences, maxlen=max_length, padding='post', truncating='post')

training_padded = np.array(training_padded)
training_labels = np.array(training_clickbait)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_clickbait)



model = Sequential()
model.add(Embedding(
    numUniqueWords, embedding_dim, input_length=max_length, embeddings_initializer=Constant(embedding_matrix)))


model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True))),
model.add( tf.keras.layers.Dense(64, activation='relu'))
model.add( tf.keras.layers.Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))


#LEADS TO SUPER LOW EVALS
# model.add(tf.keras.layers.LSTM(units = 128,dropout = 0.1, recurrent_dropout = 0.1))
# model.add(tf.keras.layers.Dense(128))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.Dense(1))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Dense(1,activation='sigmoid'))


    # model.add(keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

    # model.add(Dense(64,activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())

    # model.add(Dense(1,activation='sigmoid'))

# model.add(tf.keras.layers.BatchNormalization())

# model.add(SpatialDropout1D(0.2))

# model.add(tf.keras.layers.LSTM(64,dropout = 0.2,recurrent_dropout=0.2))
# model.add(tf.keras.layers.BatchNormalization())

# model.add(Dense(1,activation='sigmoid'))

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer= tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()

num_epochs = 20
history = model.fit(x=np.array(training_padded), y=np.array(training_clickbait), shuffle=False,
                    epochs=num_epochs, validation_data=(np.array(testing_padded), np.array(testing_clickbait)), verbose=2)

model.save('tfmodels')


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")




# Try learning about
# bias
# W_regularizer
# stanford glove embedding