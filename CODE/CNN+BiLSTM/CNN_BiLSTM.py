# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 07:41:30 2020


"""

from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer,WordNetLemmatizer
from num2words import num2words
import string
import re
import os
import pandas as pd
import numpy as np
import pickle

import keras
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Concatenate, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import model_from_json

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def preProcess(document):
    #Tokenizing the document
    words=word_tokenize(document)
    
    #Removing Punctuation Marks
    punctuationMarks=set(string.punctuation)
    temp=[]
    for word in words:
        if word not in punctuationMarks:
            temp.append(word)
    words=temp

    
    #Removing Stop Words
    temp=[]
    stopWords=set(stopwords.words('english'))
    for word in words:
        if word not in stopWords:
            temp.append(word)
    words=temp
    
    """
    #Performing stemming
    stemmer=PorterStemmer()
    sz=len(words)
    for k in range(sz):
        words[k]=stemmer.stem(words[k])
    """
        
    #Performing lemmatization
    lemmatizer = WordNetLemmatizer()
    sz=len(words)
    for k in range(sz):
        words[k]=lemmatizer.lemmatize(words[k])
    
    #Converting number to words
    sz=len(words)
    for k in range(sz):
         obj=re.match(r'^[0-9,]+$',words[k])
         if obj:
            words[k]=words[k].replace(",","")
            words[k]=num2words(int(words[k]))
    
    
    return words


classContent={}

#Reading the contents of the documents from specified directories
def readAllFiles(directoryName,truthLabels):
    global classContent
    contents=os.listdir(directoryName)
    item=0
    for current in contents:
        f = open(directoryName+"/"+current,"r",encoding="utf8")
        cIndex=truthLabels.iloc[item,1]
        classContent[cIndex].append([f.read(),cIndex])
        f.close()
        item+=1

"""
#Output Values
0 - Congress
1 - Neutal
2 - BJP
"""

"""
outputValues=[0,1,2]

for val in outputValues:
    classContent[val]=[]

newTruthDf=pd.read_excel(r'E:/Gowtham/IIITD Semesters/Semester2/Information Retrieval/Project/Dataset/NewGroundTruth.xlsx')
newTruthDf.iloc[:,1]+=1

oldP1TruthDf=pd.read_excel(r'E:/Gowtham/IIITD Semesters/Semester2/Information Retrieval/Project/Dataset/OldGroundTruth(75-270).xlsx')
oldP1TruthDf.iloc[:,1]+=1

oldP2TruthDf=pd.read_excel(r'E:/Gowtham/IIITD Semesters/Semester2/Information Retrieval/Project/Dataset/OldGroundTruth(Rest).xlsx')
oldP2TruthDf.iloc[:,1]+=1


parentPath='E:/Gowtham/IIITD Semesters/Semester2/Information Retrieval/Project/Dataset'

readAllFiles(parentPath+'/OldNewsArticle(P1)',oldP1TruthDf)
readAllFiles(parentPath+'/OldNewsArticle(P2)',oldP2TruthDf)
readAllFiles(parentPath+'/NewNewsArticle',newTruthDf)

trainRatio=0.80
testRatio=0.10
trainData=[]
validationData=[]
testData=[]
YTrain=[]
YTest=[]
YValidate=[]

itemNum=0
testDataIndex=[]
validationDataIndex=[]
for key,value in classContent.items():
    numDocs=len(value)
    trSz=int(trainRatio*numDocs)
    tSz=int(testRatio*numDocs)
    trainData.extend(classContent[key][0:trSz])
    itemNum+=trSz
    #YTrain.extend([key]*trSz)
    testData.extend(classContent[key][trSz:trSz+tSz])
    testDataIndex.append([itemNum,itemNum+tSz])
    itemNum+=tSz
    #YTest.extend([key]*tSz)
    #YValidate.extend([key]*(numDocs-(trSz+tSz)))
    validationData.extend(classContent[key][trSz+tSz:])
    validationDataIndex.append([itemNum,itemNum+(numDocs-(trSz+tSz))])
    itemNum+=(numDocs-(trSz+tSz))

trainSize=len(trainData)
validationSize=len(validationData)
testSize=len(testData)

for i in range(trainSize):
    YTrain.append(trainData[i][1])
    trainData[i]=trainData[i][0]

for i in range(testSize):
    YTest.append(testData[i][1])
    testData[i]=testData[i][0]
    
for i in range(validationSize):
    YValidate.append(validationData[i][1])
    validationData[i]=validationData[i][0]

taggedData=[TaggedDocument(words=preProcess(docData.lower()), tags=[str(i)]) for i,docData in enumerate(trainData)]

numEpochs=100
vectorSize=300
alpha=0.025

#Distributed memory model
model1=Doc2Vec(vector_size=vectorSize,alpha=alpha,min_alpha=0.00025,
              min_count=1,dm=1)
model1.build_vocab(taggedData)


#Distributed bag of words model
model2=Doc2Vec(vector_size=vectorSize,alpha=alpha,min_alpha=0.00025,
              min_count=1,dm=0)
model2.build_vocab(taggedData)


for epoch in range(numEpochs):
    #Training the model
    model1.train(taggedData,total_examples=model1.corpus_count,epochs=model1.epochs)
    model2.train(taggedData,total_examples=model2.corpus_count,epochs=model2.epochs)
    
    model1.alpha-=0.0002
    model2.alpha-=0.0002
    
    model1.min_alpha=model1.alpha
    model2.min_alpha=model2.alpha
    
model1.save("model1.model")
model2.save("model2.model")

#Using trained Doc2Vec model
model1=Doc2Vec.load("model1.model")
model2=Doc2Vec.load("model2.model")


shp1=model1.docvecs[0].shape[0]
shp2=model2.docvecs[0].shape[0]

XTrain=np.zeros([trainSize,shp1+shp2,1])
for i in range(trainSize):
    XTrain[i]=np.concatenate((model1.docvecs[i],model2.docvecs[i]),axis=None).reshape((shp1+shp2,1))
    
XValidate=np.zeros([validationSize,shp1+shp2,1])
for i in range(validationSize):
    validationSample=preProcess(validationData[i].lower())
    v1=model1.infer_vector(validationSample)
    v2=model2.infer_vector(validationSample)
    XValidate[i]=np.concatenate((v1,v2),axis=None).reshape((shp1+shp2,1))

XTest=np.zeros([testSize,shp1+shp2,1])
for i in range(testSize):
    testSample=preProcess(testData[i].lower())
    v1=model1.infer_vector(testSample)
    v2=model2.infer_vector(testSample)
    XTest[i]=np.concatenate((v1,v2),axis=None).reshape((shp1+shp2,1))


#Storing the document vector of all documents in pickle file
parentPath='E:/Gowtham/IIITD Semesters/Semester2/Information Retrieval/Project'
pickleFile=open(parentPath+"/XTrain.pickle","wb")
pickle.dump(XTrain,pickleFile)
pickleFile.close()

pickleFile=open(parentPath+"/XValidate.pickle","wb")
pickle.dump(XValidate,pickleFile)
pickleFile.close()

pickleFile=open(parentPath+"/XTest.pickle","wb")
pickle.dump(XTest,pickleFile)
pickleFile.close()

pickleFile=open(parentPath+"/YTrain.pickle","wb")
pickle.dump(YTrain,pickleFile)
pickleFile.close()

pickleFile=open(parentPath+"/YValidate.pickle","wb")
pickle.dump(YValidate,pickleFile)
pickleFile.close()

pickleFile=open(parentPath+"/YTest.pickle","wb")
pickle.dump(YTest,pickleFile)
pickleFile.close()
"""

#parentPath='E:/Gowtham/IIITD Semesters/Semester2/Information Retrieval/Project'
pickleFile=open("XTrain.pickle","rb")
XTrain=pickle.load(pickleFile)
pickleFile.close()

pickleFile=open("XValidate.pickle","rb")
XValidate=pickle.load(pickleFile)
pickleFile.close()

pickleFile=open("XTest.pickle","rb")
XTest=pickle.load(pickleFile)
pickleFile.close()

pickleFile=open("YTrain.pickle","rb")
YTrain=pickle.load(pickleFile)
pickleFile.close()

pickleFile=open("YValidate.pickle","rb")
YValidate=pickle.load(pickleFile)
pickleFile.close()

pickleFile=open("YTest.pickle","rb")
YTest=pickle.load(pickleFile)
pickleFile.close()

YTrain=keras.utils.to_categorical(YTrain,num_classes=3)
YValidate=keras.utils.to_categorical(YValidate,num_classes=3)
YTest=keras.utils.to_categorical(YTest,num_classes=3)

"""
inp1=Input(shape=XTrain.shape[1:])
inp2=Input(shape=XTrain.shape[1:])
inp3=Input(shape=XTrain.shape[1:])
inp4=Input(shape=XTrain.shape[1:])

learning_rate=0.0001


conv1=Conv1D(filters=300,kernel_size=2,padding='same',activation='relu',bias_regularizer=keras.regularizers.l2(learning_rate))(inp1)
maxp1=MaxPooling1D(pool_size=2)(conv1)

conv2=Conv1D(filters=300,kernel_size=3,padding='same',activation='relu',bias_regularizer=keras.regularizers.l2(learning_rate))(inp2)
maxp2=MaxPooling1D(pool_size=2)(conv2)

conv3=Conv1D(filters=300,kernel_size=4,padding='same',activation='relu',bias_regularizer=keras.regularizers.l2(learning_rate))(inp3)
maxp3=MaxPooling1D(pool_size=2)(conv3)

conv4=Conv1D(filters=300,kernel_size=5,padding='same',activation='relu',bias_regularizer=keras.regularizers.l2(learning_rate))(inp4)
maxp4=MaxPooling1D(pool_size=2)(conv4)


bilstmInp=Concatenate()([maxp1,maxp2,maxp3,maxp4])

bilstmOut=Bidirectional(LSTM(50,dropout=0.1, recurrent_dropout=0.05,bias_regularizer=keras.regularizers.l2(learning_rate)))(bilstmInp)
dense = Dense(100, activation='relu',bias_regularizer=keras.regularizers.l2(learning_rate))(bilstmOut)
op = Dense(3, activation='softmax',bias_regularizer=keras.regularizers.l2(learning_rate))(dense)

model = Model(inputs=[inp1, inp2, inp3, inp4], outputs=op)
adam=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([XTrain,XTrain,XTrain,XTrain], YTrain,
          epochs=30, batch_size=10,validation_data=([XValidate,XValidate,XValidate,XValidate],YValidate))

#Save the trained model
modelJson = model.to_json()
with open("DLmodel.json", "w") as json_file:
    json_file.write(modelJson)
model.save_weights("DLmodel.h5")
print("Saved model to disk")
"""

#Load the trained module
jsonFile = open('DLmodel.json', 'r')
loadedModelJson = jsonFile.read()
jsonFile.close()
loadedModel = model_from_json(loadedModelJson)
loadedModel.load_weights("DLmodel.h5")
print("Loaded trained model")

YPredicted1=loadedModel.predict([XTest,XTest,XTest,XTest])
YTest=np.argmax(YTest,axis=1)
YPredicted1=np.argmax(YPredicted1,axis=1)

YPredicted2=loadedModel.predict([XValidate,XValidate,XValidate,XValidate])
YValidate=np.argmax(YValidate,axis=1)
YPredicted2=np.argmax(YPredicted2,axis=1)

YPredicted3=loadedModel.predict([XTrain,XTrain,XTrain,XTrain])
YTrain=np.argmax(YTrain,axis=1)
YPredicted3=np.argmax(YPredicted3,axis=1)


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Train Accuracy score : ",accuracy_score(YTrain,YPredicted3))
print("Validation Accuracy score : ",accuracy_score(YValidate,YPredicted2))

print("For Test set :-")
print("Accuracy score : ",accuracy_score(YTest,YPredicted1))
print("Recall score(Macro) : ",recall_score(YTest,YPredicted1,average='macro'))
print("Recall score(Micro) : ",recall_score(YTest,YPredicted1,average='micro'))
print("Precision score(Macro) : ",precision_score(YTest,YPredicted1, average='macro'))
print("Precision score(Micro) : ",precision_score(YTest,YPredicted1, average='micro'))
print("F1 score(Macro) : ",f1_score(YTest,YPredicted1, average='macro'))
print("F1 score(Micro) : ",f1_score(YTest,YPredicted1, average='micro'))

print("Confusion Matrix : \n",confusion_matrix(YTest,YPredicted1))

cf_matrix = [[13,4,3],[5,32,0],[1,1,31]]
ax=plt.subplot()
sns.heatmap(cf_matrix, annot=True,cmap='Blues')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['BJP','Neutral','Congress']); ax.yaxis.set_ticklabels(['BJP','Neutral','Congress']);

