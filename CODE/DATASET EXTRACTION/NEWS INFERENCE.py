#!/usr/bin/env python
# coding: utf-8

# In[51]:


import glob
import string
from nltk import word_tokenize 
from nltk.corpus import stopwords


# In[52]:


stop = set(stopwords.words('english'))


# In[35]:


urlFile = open("url","r").read()


# In[36]:


sources = []
sourceCount = {}
for url in urlFile.split("\n"):
    websiteName = (url.split("/"))[2]
    sources += [websiteName]
    if websiteName in sourceCount:
        sourceCount[websiteName] += 1
    else:
        sourceCount[websiteName] = 1 
        
    


# In[37]:


print("Number of sources for news : ",len(set(sources)))


# In[64]:


print(*[j[0] for j in sorted(sourceCount.items(),key = lambda x:x[1],reverse=True)][:10])


# In[39]:


print("Data date range : 19/1/2020 to 17/2/2020")


# In[66]:


articles = glob.glob("NewsArticle/*")
data = []
for files in articles:
    data += word_tokenize(open(files,"r",encoding = "utf8").read())


# In[67]:


print(len(data)/2876)
dataDict = {}
for d in data:
    d = d.lower()
    if d not in stop and d not in string.punctuation:
        if d in dataDict:
            dataDict[d] += 1
        else:
            dataDict[d] = 1


# In[63]:


print(sorted(dataDict.items(),key=lambda x:x[1],reverse = True)[:30])


# In[ ]:




