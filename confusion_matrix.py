from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
import pandas as pd

def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	# trainY = to_categorical(trainY)
	# testY = to_categorical(testY)
	return trainX, trainY, testX, testY


def conf_matrix(y_pred, y_true):
    return pd.crosstab(y_true, y_pred)


def create_confusion_matrix():
    model = load_model('model/final_model.h5')
    trainX, trainY, testX, testY = load_dataset()
    predY = model.predict(testX, verbose = 1)
    predY = np.argmax(predY, axis=1, out=None)
    y_actu = pd.Series(testY, name='Actual')
    y_pred = pd.Series(predY, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    
    return df_confusion, predY


def greatest_confusion_predicted(df):
    #print(df)
    temp = []
    for column in df.columns:
        max = df[column].max()
        sum = df[column].sum()
        conf = sum - max
        temp.append(conf)
    return temp


def greatest_confusion_actual(df):
    #print(df)
    temp = []
    for i, val in enumerate(df)) :
        row = df.loc[val, :]
        max_ = max(row)
        sum_ = sum(row)
        conf = sum_ - max_
        temp.append(conf) 

    # for i in df.index:
    #     print(i)
    return temp








