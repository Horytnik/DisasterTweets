from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D

dataSubsetTrain = pd.read_csv('./train.csv')
dataSubsetTest = pd.read_csv('./test.csv')
# print(dataSubset.head())

Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataSubsetTrain.text, dataSubsetTrain.target, test_size= 0.4286, random_state= 1)

Xtrain = dataSubsetTrain.text
Ytrain = dataSubsetTrain.target

XValid = dataSubsetTest.text
XValidIdx = dataSubsetTest.id

wordVector = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word', #You can also try 'char'
                            stop_words= 'english',ngram_range=(1,3),dtype=np.float32)
#
XtrainVect = wordVector.fit_transform(Xtrain)
XtestVect = wordVector.transform(Xtest)
#
model = LogisticRegression(random_state = 1)
model.fit(XtrainVect, Ytrain)

print(model.score(XtrainVect, Ytrain))

plt.figure()
plot_confusion_matrix(model, XtestVect, Ytest)
plt.show()

print(classification_report(Ytest, model.predict(XtestVect)))
#
# pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer()),
#                 ('best', TruncatedSVD(n_components=120)),
#                 ('logistic', LogisticRegression())
#             ])
#
# reg = pipeline.fit(XtrainVect, Ytrain)
#
# print(reg.score(XtrainVect, Ytrain))


N = 1000
tokenizer = Tokenizer(num_words=N, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(Xtrain)
word_index = tokenizer.word_index

X_train_mat = tokenizer.texts_to_sequences(Xtrain)
X_train_mat = pad_sequences(X_train_mat, maxlen=100)



model = Sequential()
model.add(Embedding(N, 128, input_length=X_train_mat.shape[1]))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

epochs = 15
batch_size = 64

history = model.fit(X_train_mat, Ytrain, epochs=epochs, batch_size=batch_size, validation_split=0.1)
print(history)

X_test_mat = tokenizer.texts_to_sequences(Xtest)
X_test_mat = pad_sequences(X_test_mat, maxlen=100)

XTestPredict = model.predict(X_test_mat)
XTestPredict = XTestPredict.round()
XTestPredict = XTestPredict.astype(int)

XTestPredict = pd.DataFrame(XTestPredict)
XTestPredict = XTestPredict.rename(columns = {0:'target'})
XTestPredict['id'] = XValidIdx
XTestPredict = XTestPredict[['id', 'target']]
XTestPredict = XTestPredict.sort_values('id')

pd.DataFrame(XTestPredict).to_csv("submission.csv", index=False)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(0.65, 1)
plt.show()
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print('end')
'''
model = Sequential()
model.add(Embedding(N, 128, input_length=X_train_mat.shape[1]))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64,kernel_size=5, activation='relu', padding="valid", strides = 1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

epochs = 50
batch_size = 64

history = model.fit(X_train_mat, Ytrain, epochs=epochs, batch_size=batch_size, validation_split=0.1)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print('end')

'''