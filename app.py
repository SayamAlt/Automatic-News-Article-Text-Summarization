#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
from transformers import pipeline 
from newspaper import Article

# In[3]:

@st.cache(allow_output_mutation=True)
def get_model():
    return pipeline("summarization",model="facebook/bart-large-cnn")

# In[6]:


def scrapArticle(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        article.nlp()
    except Exception as e:
        print(e)
    return article.text


# In[8]:


def main():
    st.title("Automatic News Articles Text Summarization")
    link = st.text_input("Enter your URL:")
    article = scrapArticle(link)
    summary = ""
    bart_model = get_model();
    
    if st.button("Summarize"):
        summary = bart_model(article)[0]['summary_text']

    st.success(summary)


# In[9]:


if __name__ == "__main__":
    main()

