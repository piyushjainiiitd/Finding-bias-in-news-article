#!/usr/bin/env python
# coding: utf-8

# In[7]:


import re
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from num2words import num2words
import contractions


# In[16]:


class preprocess:
    wordRegex = r"\S+"
    emailRegex = r"\b[\w|_|-|$|#|~]+(?:\.[\w|_|-|$|#|~]+)*@[\w|-]+(?:\.[\w|-]+)*(?:\.[A-Z|a-z]{2,})"
    websiteRegex = r"(?:http:\/\/|https:\/)??(?:\w+\.)+\w+"
    decimalRegex = r"^\d+(?:\.\d+)?$"
    
    wordTokenizer = RegexpTokenizer(wordRegex)
    emailTokenizer = RegexpTokenizer(emailRegex)
    websiteTokenizer = RegexpTokenizer(websiteRegex)
    
    emailRe = re.compile(emailRegex)
    decimalRe = re.compile(decimalRegex)
    websiteRe = re.compile(websiteRegex)
    
    punctuations = '"\'#$%&()*+,/:;<=>?[\\]^_`{|}~@.-!'
    convertPunctuations = str.maketrans(punctuations," "*len(punctuations))
    
    stopWords = set(stopwords.words('english')) 
    
    wordNet = WordNetLemmatizer()
    
    def wordSeperator(self,sentence):
        return self.wordTokenizer.tokenize(sentence)
    
    def checkEmail(self,sentence):
        return self.emailTokenizer.tokenize(sentence)
    
    def removePunctuations(self,sentence):
        return sentence.translate(self.convertPunctuations)
    
    def checkStopWord(self,word):
        return True if word in self.stopWords else False
    
    def removeEmail(self,sentence):
        return self.emailRe.sub("",sentence)
    
    def checkWebsite(self,sentence):
        return self.websiteTokenizer.tokenize(sentence)
    
    def removeWebsite(self,sentence):
        return self.websiteRe.sub("",sentence)
    
    def lemmatizeWord(self,word):
        return self.wordNet.lemmatize(self.wordNet.lemmatize(self.wordNet.lemmatize(word,pos='v'),pos ='a'),pos='n')
        
    def caseChange(self,word):
        return word.lower()
    
    def changeNumber(self,word):
        if not(word.isdecimal()):
            return [word]
        else:
            words = ((num2words(int(word))+"").translate(self.convertPunctuations)).split()
            wordList = []
            for word in words:
                wordList += word.split("-")
            if "and" in wordList:
                wordList.remove("and")
            return wordList
    
    def contractionsExpand(self,sentence):
        return contractions.fix(sentence,slang=False)


# In[17]:



# In[ ]:





# In[ ]:




