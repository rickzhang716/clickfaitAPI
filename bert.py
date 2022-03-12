from distutils.errors import PreprocessError
import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import pandas as pd
import numpy as np

import matplotlib as plt
from matplotlib import pyplot as pyplt
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report


training_size = 25000
num_epochs = 10




columnNames = ["headline", "clickbait"]
clickbait_dataset = pd.read_csv('datasets/randomized_dataset1.csv')
inputTitles = clickbait_dataset.headline.to_list()
clickbait = clickbait_dataset.clickbait.to_list()


training_sentences = inputTitles[0:training_size]
testing_sentences = inputTitles[training_size:30000]
training_clickbait = clickbait[0:training_size]
testing_clickbait = clickbait[training_size:30000]



# print(clickbait_dataset.groupby('clickbait').describe())
# for i in range(5):
#     print(training_sentences[i])


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder= hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


def get_sentence_embedding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']



#bert layers
text_input = tf.keras.layers.Input(shape=(),dtype=tf.string, name ="text")
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

#nn layers
out = tf.keras.layers.Dropout(0.1,name="dropout")(outputs['pooled_output'])
out = tf.keras.layers.Dense(1, activation = 'sigmoid',name = 'output')(out)

#final model

model = tf.keras.Model(inputs = [text_input], outputs = out)
model.summary()

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',loss='binary_crossentropy',metrics = METRICS)

model.fit(training_sentences,training_clickbait,epochs = num_epochs)


model.evaluate(testing_sentences,testing_clickbait)
model.save('tfmodels2')

model.predict(["The spurs win 99-96 after buzzer beater from Kawhi Leonard","Find out what type of tree you are"])

predicted = model.predict(testing_sentences)
np.savetxt("ans.txt",predicted)

predicted = predicted.flatten()
np.savetxt("flat.txt",predicted)

predicted = np.where(predicted >0.5,1,0)
np.savetxt("simplified.txt",predicted)

matrix = confusion_matrix(testing_clickbait,predicted)
sn.heatmap(matrix,annot=True,fmt='d')
pyplt.xlabel('Predicted')
pyplt.ylabel('Truth')

