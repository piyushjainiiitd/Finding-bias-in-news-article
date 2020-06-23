#!/usr/bin/env python
# coding: utf-8

# In[71]:


# Used news api to extract news URL 
from newsapi import NewsApiClient
from newspaper import Article
import time
import datetime
from tqdm import tqdm as tqdm
import random


# In[72]:


newsapi = NewsApiClient(api_key='d4a46de48de7449a974bd7c8fb2d87af') # Connect to news api.org


# In[73]:


url = [] # Url list
today = datetime.date.today() # GEt today date
for source in tqdm(range(30)): # Iterate over last 30 days as free version only support these date range
    for p in range(1,6): # Maximum 5 pages or 100 articles a day
        all_articles = newsapi.get_everything(q='caa',language='en',sort_by='relevancy',page=p,from_param=str(today),to=str(today)) # Get News article infromation 20 per iteration
        for i in all_articles['articles']: # For all 20 articles
            url+=[i['url']] # Extract article url
    today -= datetime.timedelta(days=1) # Decrease date by 1 day


# In[74]:


url.sort() # Sort the url by name 
random.shuffle(url) # Shuffle so that one website is not hit consecutively


# In[78]:


article = [] # For article 
count = 0 # Get count
for index,value in enumerate(url):
    try:
        a = Article(value) # Load article url
        a.download() # Download article
        a.parse() # Parse so that http tags are differentiated from text of article
        article.append(a.text) # append article tag
        time.sleep(10) # Sleep for 10 second before other article to avoid getting blocked
        open("NewsArticle/"+str(index),"w",encoding="utf8").write(a.text) # Store news article in NewsArticle Folder
    except:
        pass # Pass this iteration 
    


# In[ ]:


open("url","w",encoding = "utf8").write("\n".join(url)) # Append url 

