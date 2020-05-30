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
from sklearn.model_selection import cross_val_score

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Embedding,Flatten, Dropout, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping

import functions

# Reading data
dataSubsetTrain = pd.read_csv('./train.csv')
dataSubsetTest = pd.read_csv('./test.csv')

#Splitting by target
target0 = dataSubsetTrain.text[dataSubsetTrain['target'] == 0]
target1 = dataSubsetTrain.text[dataSubsetTrain['target'] == 1]

dataSubsetTrain = pd.read_csv('./train.csv')

# get frequency of all words in  text
dictTar0 = functions.word_frequency(target0)
dictTar1 = functions.word_frequency(target1)

dictTarRemStop0 = functions.word_frex_without_stop(target0)
dictTarRemStop1 = functions.word_frex_without_stop(target1)


# Plot the most frequent words
plt.figure()
y_pos = np.arange(len(dictTarRemStop0.word.head(50)))
x_pos = dictTarRemStop0.wordcount.head(50)
plt.barh(y_pos, x_pos)
plt.title('Frequency without stopwords when it is not accident')
plt.yticks(y_pos, dictTarRemStop0.word.head(50) )
plt.gca().invert_yaxis()
plt.show()

plt.figure()
y_pos = np.arange(len(dictTarRemStop1.word.head(50)))
x_pos = dictTarRemStop1.wordcount.head(50)
plt.barh(y_pos, x_pos)
plt.title('Frequency without stopwords when it is accident')

plt.yticks(y_pos, dictTarRemStop1.word.head(50) )

plt.gca().invert_yaxis()
plt.show()

# Split data set into train and test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataSubsetTrain.text, dataSubsetTrain.target, test_size= 0.4286, random_state= 1)

Xtrain = dataSubsetTrain.text
Ytrain = dataSubsetTrain.target

XValid = dataSubsetTest.text
XValidIdx = dataSubsetTest.id

# TFIDF vectorizer
wordVector = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word', #You can also try 'char'
                            stop_words= 'english',ngram_range=(1,3),dtype=np.float32)
XtrainVect = wordVector.fit_transform(Xtrain)
XtestVect = wordVector.transform(Xtest)

# Logistic Regression model
model = LogisticRegression(random_state = 1)
model.fit(XtrainVect, Ytrain)

print(model.score(XtrainVect, Ytrain))

# Confusion matrix
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

# Removing stopwords and unneccesary words
XtrainRvStop = functions.remove_stop_words(Xtrain)

N = 180
# Tokenization
tokenizer = Tokenizer(num_words=N, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(XtrainRvStop)
word_index_train = tokenizer.word_index

X_train_mat = tokenizer.texts_to_sequences(XtrainRvStop)
X_train_mat = pad_sequences(X_train_mat, maxlen=100)

# Sequential neural network model
def sequential_model(optimizer):
    model = Sequential()
    model.add(Embedding(N, 128, input_length=X_train_mat.shape[1]))
    # model.add(Flatten())
    model.add(Conv1D(128, 5, activation='relu'))
    # model.add(GlobalMaxPooling1D())
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
    return model

# Grid search for parameters
# classifier = KerasClassifier(build_fn=sequential_model)
# parameters = {'batch_size': [40,45],
#               'epochs': [10,12],
#               'optimizer': ['adam', 'rmsprop'],
#               # 'dropout1' : [0.2,0.25,3],
#               # 'dropout2' : [0.2,0.25,3],
#               }
# grid_search = GridSearchCV(estimator=classifier,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10)
# grid_search = grid_search.fit(X_train_mat, Ytrain)

# Training the model
epochs = 7
batch_size = 45
model = sequential_model('adam')
history = model.fit(X_train_mat, Ytrain,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2)

# Prediction for kaggle score
XValidRvStop = functions.remove_stop_words(XValid)

X_valid_mat = tokenizer.texts_to_sequences(XValidRvStop)
X_valid_mat = pad_sequences(X_valid_mat, maxlen=100)

XValidPredict = model.predict(X_valid_mat)
XValidPredict = XValidPredict.round()
XValidPredict = XValidPredict.astype(int)

XValidPredict = pd.DataFrame(XValidPredict)
XValidPredict = XValidPredict.rename(columns = {0:'target'})
XValidPredict['id'] = XValidIdx
XValidPredict = XValidPredict[['id', 'target']]
XValidPredict = XValidPredict.sort_values('id')

pd.DataFrame(XValidPredict).to_csv("NNsubmission.csv", index=False)

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(0.5, 1)
# plt.ylim(0, 1)
plt.show()

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.ylim(0, 0.9)
plt.ylim(0, 1)
plt.show()
print('end')
