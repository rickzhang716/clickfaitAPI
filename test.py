import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json

import numpy as np
from numpy.core.fromnumeric import squeeze

import pandas as pd

def predict(text):
    mySequence = tokenizer.texts_to_sequences(text)
    mySeqPadded = pad_sequences(
        mySequence, maxlen=max_length, padding='post', truncating='post')
    ans = model.predict(mySeqPadded)
    return ans


vocab_size = 10000
embedding_dim = 50
max_length = 15
training_size = 25000

new_model = tf.keras.models.load_model('tfmodels')
new_model.summary()
model = new_model


clickbait_dataset = pd.read_csv('datasets/clickbait_data.csv')

inputTitles = clickbait_dataset.headline.to_list()
clickbait = clickbait_dataset.clickbait.to_list()


tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

tokenizer.fit_on_texts(clickbait_dataset)
word_index = tokenizer.word_index

f = open('ans.csv','w')

for i in range(30000,32000):
    text = [inputTitles[i]]
    print(i)
    

  
    # text = ["I am going to buy a new car"]
    res = predict(text)

    # f.write(inputTitles[i] + "          "=  + str(ans) +f"/{clickbait[i]}\n")
    f.write(f"{inputTitles[i]}         {res}/{clickbait[i]} \n")
f.close()

print(predict("asga"))
