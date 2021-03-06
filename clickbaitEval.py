
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json

import numpy as np
from numpy.core.fromnumeric import squeeze

import matplotlib.pyplot as plt

from google.cloud import language

import pandas as pd

from flask import Flask
from flask import request
from flask import jsonify
from flask.wrappers import Request




app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


def analyze_text_sentiment(text):
    client = language.LanguageServiceClient()
    document = language.Document(
        content=text, type_=language.Document.Type.PLAIN_TEXT)

    response = client.analyze_sentiment(document=document)

    sentiment = response.document_sentiment
    results = dict(
        text=text,
        score=f"{sentiment.score:.1}",
        magnitude=f"{sentiment.magnitude:.1}",
    )
    return results['score']

vocab_size = 10000
embedding_dim = 30
max_length = 10
training_size = 25000


# columnNames = ["headline", "clickbait"]
# df = pd.read_csv(
#     'api/clickbait_data.csv')

# inputTitles = df.headline.to_list()
# clickbait = df.clickbait.to_list()


# training_sentences = inputTitles[0:training_size]
# testing_sentences = inputTitles[training_size:]
# training_clickbait = clickbait[0:training_size]
# testing_clickbait = clickbait[training_size:]

dataset = pd.read_csv("datasets/clickbait_data.csv")

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

tokenizer.fit_on_texts(dataset)
word_index = tokenizer.word_index


# training_sequences = tokenizer.texts_to_sequences(training_sentences)
# training_padded = pad_sequences(
#     training_sequences, maxlen=max_length, padding='post', truncating='post')


# testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
# testing_padded = pad_sequences(
#     testing_sequences, maxlen=max_length, padding='post', truncating='post')


# training_padded = np.array(training_padded)
# training_labels = np.array(training_clickbait)
# testing_padded = np.array(testing_padded)
# testing_labels = np.array(testing_clickbait)


# model = tf.keras.Sequential(tf.keras.layers.Embedding(
#     vocab_size, embedding_dim, input_length=max_length))
# model.add(tf.keras.layers.Convolution1D(32, 2))

# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.Convolution1D(32, 2))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation('relu'))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(1))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation('sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam', metrics=['accuracy'])
# model.summary()

# num_epochs = 20
# history = model.fit(x=np.array(training_padded), y=np.array(training_clickbait), shuffle=True,
#                     epochs=num_epochs, validation_data=(np.array(testing_padded), np.array(testing_clickbait)), verbose=2)

# model.save('tfmodels')


# def plot_graphs(history, string):
#     plt.plot(history.history[string])
#     plt.plot(history.history['val_'+string])
#     plt.xlabel("Epochs")
#     plt.ylabel(string)
#     plt.legend([string, 'val_'+string])
#     plt.show()


# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")

new_model = tf.keras.models.load_model('tfmodels')
new_model.summary()
model = new_model

@app.route("/go", methods = ['GET','POST'])
def main():
    print(request.json["title"])
    text = []
    text.append(request.json["title"])

  
    # text = ["I am going to buy a new car"]
    mySequence = tokenizer.texts_to_sequences(text)
    mySeqPadded = pad_sequences(
        mySequence, maxlen=max_length, padding='post', truncating='post')
    # print(model.predict(mySeqPadded)[0, 0])
    ans = model.predict(mySeqPadded)
    ans = ans*100
    sentiment =  abs(float(analyze_text_sentiment(text[0])))
    print(sentiment)
    if ans>50:
        print("true")
        ans = str(ans)
        answers = ans.strip("[] ")
        response = {"clickbait":answers,"sentiment":sentiment}
        print(response)
    else:
        ans = str(ans)
        answers = ans.strip("[] ")
        response = jsonify(clickbait=answers,sentiment=-1)
    
    # print(answers)
    return response
    # return ("{:0.2f}".format(ans))


