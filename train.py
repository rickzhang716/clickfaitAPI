
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from keras_preprocessing.text import tokenizer_from_json
from numpy.core.fromnumeric import squeeze

vocab_size = 10000
embedding_dim = 30
max_length = 10
training_size = 25000


columnNames = ["headline", "clickbait"]
df = pd.read_csv(
    'datasets/clickbait_data.csv')

clickbait_dataset = df.sample(frac=1)
clickbait_dataset.to_csv("datasets/randomized_dataset.csv")
inputTitles = clickbait_dataset.headline.to_list()
clickbait = clickbait_dataset.clickbait.to_list()


training_sentences = inputTitles[0:training_size]
testing_sentences = inputTitles[training_size:]
training_clickbait = clickbait[0:training_size]
testing_clickbait = clickbait[training_size:]


tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index


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


model = tf.keras.Sequential(tf.keras.layers.Embedding(
    vocab_size, embedding_dim, input_length=max_length))
model.add(tf.keras.layers.Convolution1D(32, 2))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Convolution1D(32, 2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 20
history = model.fit(x=np.array(training_padded), y=np.array(training_clickbait), shuffle=True,
                    epochs=num_epochs, validation_data=(np.array(testing_padded), np.array(testing_clickbait)), verbose=2)

model.save('tfmodels')


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")