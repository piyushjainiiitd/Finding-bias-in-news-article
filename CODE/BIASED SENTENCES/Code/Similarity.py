import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import math
import operator
import nltk
import string
import os
from sklearn.model_selection import train_test_split
import re
import num2words

def preprocess(s):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(s)
    filter_tokens=[]
    for i in tokens:
        if i not in stop_words and i not in string.punctuation:
            filter_tokens.append(i)
    for i in range(len(filter_tokens)):                 
        try:
            if re.search("(\d+)",filter_tokens[i]):     
                filter_tokens[i] = filter_tokens[i].replace(',','')
                filter_tokens[i] = num2words(filter_tokens[i])
        except:
            pass 
    return filter_tokens


def result(test):                         
    values = {}                                         
    for doc in range(len(X_train)):
        temp = np.dot(X_train[doc], test)      
        temp = np.dot(temp, temp)
        numer = math.sqrt(np.sum(temp))
        d1 = np.dot(test,test)
        d1 = np.sqrt(np.sum(d1))
        d2 = np.dot(X_train[doc], X_train[doc])
        d2 = np.sqrt(np.sum(d2))
        try:
            values[doc] = numer/(d1*d2)
        except:
            values[doc] = 0
    val = max(values.items(), key = operator.itemgetter(1))[0] 
    return Y_train[val]
    
def testDocSentence():
    path = "E:\\Semester II\\IR\\Project\\Data\\Dup"
    d = {0:[], -1:[], 1:[]}
    for file in os.listdir(path):
        f = open(path+"\\"+file, 'r', encoding='utf-8')
        text = f.read()
        sent_text = nltk.sent_tokenize(text)
        to_test = []
        for i in sent_text:
            to_test.append(preprocess(i))
        testData = []
        for data in range(len(to_test)):
            testData.append(model.infer_vector(to_test[data]))
        for i in range(len(testData)):
            val = result(testData[i])
            d[val].append(sent_text[i])
    return d


def accuracy(X,Y,s):
    c=0
    for test in range(len(X)):
        k = result(X[test])
        if k==Y[test]:
            c+=1
    print(s, " Accuracy: ", (c/len(X))*100)

df = pd.read_csv("E:\\Semester II\\IR\\Project\\FinalAnnotationData.txt", header=None, delimiter='\t')
X_train = list(df[1])
Y_train = list(df[0])
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state = 42)
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(X_train)]
X = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(X_test)]
max_epochs = 100
vec_size = 150
alpha = 0.0025
model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
model.build_vocab(tagged_data)
for epoch in range(max_epochs):
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    model.alpha -= 0.00002
    model.min_alpha = model.alpha

X_train = []
X_test = []
for data in range(len(X)):
    X_test.append(model.infer_vector(X[data][0]))
for data in range(len(tagged_data)):
    X_train.append(model.infer_vector(tagged_data[data][0]))
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
accuracy(X_test, Y_test, "Test")
print(testDocSentence())
