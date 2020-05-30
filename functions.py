import pandas as pd
import numpy as np


from wordcloud import STOPWORDS

dumblist = ['co', 'amp', '&amp', 'will',' ','-','', 'via', '??' ]

# Function which calculates frequency of words
def word_frequency(inputText):

    freqDict = {}

    for sentence in inputText:
        sentence = sentence.split(' ')

        for word in sentence:
            if word in freqDict:
                freqDict[word] = freqDict[word]+1
            else:
                freqDict[word] = 1
    freqDict = pd.DataFrame(sorted(freqDict.items(), key=lambda x: x[1], reverse = True))
    freqDict.columns = ["word", "wordcount"]
    return freqDict

# Function which calculates frequency of words but with removed stopwords.
def word_frex_without_stop(inputText):
    freqDict = {}

    for sentence in inputText:
        sentence = sentence.lower()
        sentence = sentence.split(' ')
        for word in sentence:
            if (word not in STOPWORDS) & (word not in dumblist) & (not word.startswith( 'http' )) & (not word.startswith( 'https' ))& (not word.startswith( '&amp' )):
                if (word in freqDict):
                    freqDict[word] = freqDict[word]+1
                else:
                    freqDict[word] = 1
    freqDict = pd.DataFrame(sorted(freqDict.items(), key=lambda x: x[1], reverse = True))
    freqDict.columns = ["word", "wordcount"]
    return freqDict

# Function which removes stop words and unneccesary words
def remove_stop_words(inputText):

    list = []
    array = np.array([])
    for sentence in inputText:
        sentence = sentence.lower()
        sentence = sentence.split(' ')
        for word in sentence:
            if (word not in STOPWORDS) & (word not in dumblist) & (not word.startswith( 'http' )) & (not word.startswith( 'https' ))& (not word.startswith( '&amp' )):
                list.append(word)
        list = " ".join(list)
        array = np.append(array,list)
        list = []
    new_series = pd.Series(array)
    return new_series

