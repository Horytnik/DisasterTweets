import pandas as pd
import plotly.graph_objs as go

from wordcloud import STOPWORDS

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

def word_frex_without_stop(inputText):
    freqDict = {}

    for sentence in inputText:
        sentence = sentence.lower()
        sentence = sentence.split(' ')
        for word in sentence:
            if word not in STOPWORDS:
                if (word in freqDict):
                    freqDict[word] = freqDict[word]+1
                else:
                    freqDict[word] = 1
    freqDict = pd.DataFrame(sorted(freqDict.items(), key=lambda x: x[1], reverse = True))
    freqDict.columns = ["word", "wordcount"]
    return freqDict
print('aaa')
