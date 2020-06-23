#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import files
from Preprocessing import preprocess
import glob
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder


# In[2]:


preprocessing = preprocess() # Load preprocess class from preprocessing module


# In[3]:


def generateWordList(fileData): # Generate list of preprocessed words from file
    finalSet = preprocessing.checkEmail(fileData) # Check emails in the file and add them to wordlist
    fileData = preprocessing.removeEmail(fileData) # Remove emails 
    finalSet = finalSet + preprocessing.checkWebsite(fileData) # Check for website and decimal number and add them to wordlist
    fileData = preprocessing.removeWebsite(fileData) # Remove webiste and decimal numbers 
    fileData = preprocessing.contractionsExpand(fileData) # Expand the contracted word
    fileData = preprocessing.caseChange(fileData) # Change case of file to lower case
    fileData = preprocessing.removePunctuations(fileData) # Remove punctutation 
    wordList = preprocessing.wordSeperator(fileData) # Seperate words by space
    for word in wordList: # For each words in word list
        if word.isdecimal(): # If number is present expand it 
            finalSet += preprocessing.changeNumber(word) # Generate number form of the number and add it to final list 
        else: 
            finalSet.append(preprocessing.lemmatizeWord(word)) # If it is word apply lemmatization
    return finalSet # Return the fnal word list


# In[4]:


fileData = open("Annotated-Data.txt",encoding="utf8").read() # Load annotated text
count = 0 # Load count -- not used now
documents = [] # Load preprocessed documents
y = [] # Load annotation values
for i in fileData.split("\n"): # Iterate for all files 
    splitData = i.split("\t") # Annotation and text are space seperated
    if len(splitData)>1:
        y.append(int(splitData[0])) # Append value for y
        documents.append(" ".join(generateWordList(" ".join(splitData[1:])))) # Document append file
        count += 1


# In[5]:


# """***Semeval Data***"""
# fileData = open("SemEval-2019.txt",encoding="utf8").read()
# count = 0 
# documents = []
# y = []
# for i in fileData.split("\n"):
#     splitData = i.split(" ")
#     if len(splitData)>1:
#         y.append(int(splitData[1]))
#         documents.append(" ".join(generateWordList(" ".join(splitData[2:]))))
#         count += 1


# In[6]:


label = LabelEncoder() # Encode y label
label.fit(y) # Fit label
y = label.transform(y) # Transform the label


# In[7]:


Tfidf_vect = TfidfVectorizer(max_features=1000) # Used TFIDF for feature vector
Tfidf_vect.fit(documents) # Fit documents into tfidf
X = Tfidf_vect.transform(documents) # Transform the document
xTrain, xTest, yTrain, yTest = train_test_split(documents,y, test_size=0.33) # Divide train and test set


# In[8]:


trainTfidf = Tfidf_vect.transform(xTrain) # Fit train TFIDF vector
testTfidf = Tfidf_vect.transform(xTest) # Fit test TFIDF vector


# In[9]:


Naive = naive_bayes.MultinomialNB() # Create multinomial Naive bayes
Naive.fit(trainTfidf,yTrain) # Fit TFIDF vector
prediction1 = Naive.predict(testTfidf) # Predict on test set
print("Naive Bayes Accuracy : ",accuracy_score(prediction1, yTest)*100) # Print accuracy of naive bayes


# In[10]:


SVM = svm.SVC(C=0.01, kernel="poly", degree=3, gamma=10) # Run SVM with following setting
cv_results = cross_validate(SVM,X, y, cv=10,scoring=["recall_micro","precision_micro","f1_micro","accuracy","recall_macro","precision_macro","f1_macro"]) # Cross validate over 10 iterations


# In[11]:


print("Precision micro SVM : ",sum(cv_results["test_precision_micro"])/10)
print("Recall micro SVM : ",sum(cv_results["test_recall_micro"])/10)
print("F1 micro SVM : ",sum(cv_results["test_f1_micro"])/10)
print("Accuracy SVM : ",sum(cv_results["test_accuracy"])/10)
print("Precision macro SVM : ",sum(cv_results["test_precision_macro"])/10)
print("Recall macro SVM : ",sum(cv_results["test_recall_macro"])/10)
print("F1 macro SVM : ",sum(cv_results["test_f1_macro"])/10)


# In[12]:


def SVMWithSetting(C=1,kernel="linear",degree=3,gamma="auto"): # Function to run SVM with different setting
    SVM = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma) # Creat SVM Model with 
    SVM.fit(trainTfidf,yTrain) # Fit with TFIDF vector
    prediction2 = SVM.predict(testTfidf) 
    return accuracy_score(prediction2, yTest)*100 # Return accuracy score


# In[13]:


# This will generate Accuracy against different value of GAMMA and C
cValues = [0.001,0.01,0.1,1,10,100,1000]
gammaValues = [0.001,0.01,0.1,1,10,100,1000]

fig = go.Figure()
for c in cValues:
    accuracy = []
    for gamma in gammaValues:
        accuracy.append(SVMWithSetting(C=c,gamma=gamma,kernel="poly"))
    fig.add_trace(go.Scatter(x=gammaValues,y=accuracy,mode='markers+lines',name="C="+str(c)))


# In[14]:


fig.update_layout(title = "Gamma vs Accuracy for different value of C polynomial kernel",xaxis_title = "Gamma",xaxis_type = "category",yaxis_title = "accuracy") # Update layout xaxis and yaxis title
fig.show()


# In[15]:


# This will generate Accuracy against different value of GAMMA and Kernel type
shapes = [ "linear", "poly", "rbf", "sigmoid"]
gammaValues = [0.001,0.01,0.1,1,10,100,1000]

fig = go.Figure()
for shape in shapes:
    accuracy = []
    for gamma in gammaValues:
        accuracy.append(SVMWithSetting(kernel=shape,gamma=gamma))
    fig.add_trace(go.Scatter(x=gammaValues,y=accuracy,mode='markers+lines',name="kernel="+shape))


# In[16]:


fig.update_layout(title = "Gamma vs Accuracy for different kernel",xaxis_title = "Gamma",xaxis_type = "category",yaxis_title = "accuracy") # Update layout xaxis and yaxis title
fig.show()


# In[17]:


# This will generate Accuracy against different value of degree for polynomial kernel
shapes = ["poly"]
degree = [1,3,5,7,9,11]

fig = go.Figure()
for shape in shapes:
    accuracy = []
    for d in degree:
        accuracy.append(SVMWithSetting(kernel=shape,degree=d,C=0.01,gamma=10))
    fig.add_trace(go.Scatter(x=degree,y=accuracy,mode='markers+lines',name="kernel="+shape))


# In[18]:


fig.update_layout(title = "Degree of Polynomial vs Accuracy",xaxis_title = "Degree",xaxis_type = "category",yaxis_title = "accuracy") # Update layout xaxis and yaxis title
fig.show()


# In[ ]:




