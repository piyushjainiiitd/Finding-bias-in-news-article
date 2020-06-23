from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, LSTM
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.optimizers import Adam
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import os
import string
import matplotlib.pyplot as plt
import re
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import num2words

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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

def predict(data):
    v0 = []
    v1 = []
    v2 = []
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    data=np.reshape(data,(data.shape[0],1,data.shape[1]))
    predictions = classifier.predict_classes(data)
    for i in range(len(data)):
        t = predictions[i]
        if t == 0:
            v0.append(i)
        elif t==1:
            v1.append(i)
        else:
            v2.append(i)
    return v0, v1, v2

def testDocSentence():
    path = "E:\\Semester II\\IR\\Project\\Data\\Dup"
    d = {0:[], 1:[], 2:[]}
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
        v0, v1, v2 = predict(testData)
        for i in range(len(v0)):
            d[0].append(sent_text[v0[i]])
        for i in range(len(v1)):
            d[1].append(sent_text[v1[i]])
        for i in range(len(v2)):
            d[2].append(sent_text[v2[i]])
    return d
        
def TrainingData(path):
    df = pd.read_csv(path, header=None, delimiter='\t')
    Y_train = []
    for i in range(len(df)):
        t = df[0][i]
        if t==0:
            Y_train.append([0, 1, 0])
        elif t==1:
            Y_train.append([0,0,1])
        else:
            Y_train.append([1,0,0])
    X_train = list(df[1])
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state = 42)
    tagged_data = [TaggedDocument(words=nltk.word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(X_train)]
    tagged_data_test = [TaggedDocument(words=nltk.word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(X_test)]
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        model.train(tagged_data, total_examples=model.corpus_count,epochs=model.iter)
        model.alpha -= 0.00002
        model.min_alpha = model.alpha    
    X_train = []
    X_test = []
    for data in range(len(tagged_data)):
        X_train.append(model.infer_vector(tagged_data[data][0]))
    for data in range(len(tagged_data_test)):
        X_test.append(model.infer_vector(tagged_data_test[data][0]))
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_train=np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    X_test=np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
    return X_train, Y_train, X_train, Y_train

def testAccuracy(X_test, Y_test):
    r = classifier.predict_classes(X_test)
    true = []
    pred = []
    for i in range(len(Y_test)):
        if(Y_test[i][0]==1):
            true.append(0)
        elif Y_test[i][1]==1:
            true.append(1)
        else:
            true.append(2)
        pred.append(r[i])
    print(confusion_matrix(true, r))
    print("Accuracy")
    print(accuracy_score(true , r))
    print("Report")
    print(classification_report(true , r))

path = "E:\\Semester II\\IR\\Project\\FinalAnnotationData.txt"
max_epochs = 100
vec_size = 150
alpha = 0.0025
model = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm =1)
X_train, Y_train, X_test, Y_test = TrainingData(path)
classifier = Sequential()
classifier.add(LSTM(512,return_sequences=False,input_shape=(1,150)))
classifier.add(Dropout(0.2))
classifier.add(Dense(3, activation = 'softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
h = classifier.fit(X_train,Y_train, epochs=max_epochs,batch_size=32, verbose=1)
plt.plot(h.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
result = testDocSentence()
accuracy = testAccuracy(X_test, Y_test)
print(result)
