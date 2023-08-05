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
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix



new_model = tf.keras.models.load_model('tfmodels2')
new_model.summary()
model = new_model


print(model.predict(["find out what plant you are", "How the Russian government led to the rise of Putin"]))
clickbait_dataset = pd.read_csv('datasets/randomized_dataset1.csv')

inputTitles = clickbait_dataset.headline.to_list()
clickbait = clickbait_dataset.clickbait.to_list()


testing_sentences = inputTitles[30000:]
testing_clickbait = clickbait[30000:]


predicted = model.predict(testing_sentences)
print(predicted)
predicted = predicted.flatten()
predicted = np.where(predicted >0.5,1,0)

print(type(predicted))
cm = confusion_matrix(testing_clickbait,predicted)


sn.heatmap(cm,annot=True,fmt='d')
pyplt.xlabel('Predicted')
pyplt.ylabel('Truth')

pyplt.show()


# predicted = np.where(predicted > 0.5,1,0)

# print(predicted)
# predicted = predicted.flatten()

# print(predicted)


# matrix = confusion_matrix(testing_clickbait,predicted)
# sn.heatmap(matrix,annot=True,fmt='d')
# pyplt.xlabel('Predicted')
# pyplt.ylabel('Truth')




