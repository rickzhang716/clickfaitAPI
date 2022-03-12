from distutils.errors import PreprocessError
import os
import shutil
from itsdangerous import json

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import pandas as pd
import numpy as np

import matplotlib as plt
from matplotlib import pyplot as pyplt
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

from flask import Flask
from flask import request
from flask import jsonify




app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


new_model = tf.keras.models.load_model('tfmodels2')
new_model.summary()
model = new_model
def evaluate(text):
    request = [text]
    print(text)
    ans = model.predict(request)
    ans = ans.flatten()
    return ans

print(evaluate("I am going to buy ice cream today"))


@app.route("/go", methods = ['GET','POST'])
def main():
    print(request.json["title"])
    text = []
    text.append(request.json["title"])

  
    # text = ["I am going to buy a new car"]
    res = model.predict(text)
    res = res*100
    print(res)
    if res>50:
        print("true")
        res = str(res)
        answers = res.strip("[] ")
        response = jsonify(clickbait=answers,sentiment=-1)
        print(response)
    else:
        res = str(res)
        answers = res.strip("[] ")
        response = jsonify(clickbait=answers,sentiment=-1)
    
    return response
    # return ("{:0.2f}".format(ans))
# new_model = tf.keras.models.load_model('tfmodels2')
# new_model.summary()
# model = new_model


# print(model.predict(["find out what plant you are", "How the Russian government led to the rise of Putin"]))
# clickbait_dataset = pd.read_csv('datasets/randomized_dataset1.csv')

# inputTitles = clickbait_dataset.headline.to_list()
# clickbait = clickbait_dataset.clickbait.to_list()


# testing_sentences = inputTitles[30000:]
# testing_clickbait = clickbait[30000:]


# predicted = model.predict(testing_sentences)
# print(predicted)
# predicted = predicted.flatten()
# predicted = np.where(predicted >0.5,1,0)

# print(type(predicted))
# cm = confusion_matrix(testing_clickbait,predicted)


# sn.heatmap(cm,annot=True,fmt='d')
# pyplt.xlabel('Predicted')
# pyplt.ylabel('Truth')

# pyplt.show()


# predicted = np.where(predicted > 0.5,1,0)

# print(predicted)
# predicted = predicted.flatten()

# print(predicted)


# matrix = confusion_matrix(testing_clickbait,predicted)
# sn.heatmap(matrix,annot=True,fmt='d')
# pyplt.xlabel('Predicted')
# pyplt.ylabel('Truth')




