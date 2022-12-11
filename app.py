#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from transformers import pipeline 
from newspaper import Article


# In[9]:


@st.cache(allow_output_mutation=True)
def get_model():
    model = pipeline("summarization",model="t5-base")
    return model


# In[3]:


def scrapArticle(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        article.nlp()
    except Exception as e:
        print(e)
    return article.text


# In[11]:


def main():
    st.title("Automatic News Articles Text Summarization")
    link = st.text_input("Enter your URL:")
    article = scrapArticle(link)
    summary = ""
    model = get_model();
    
    if st.button("Summarize"):
        summary = model(article)[0]['summary_text']

    st.success(summary.capitalize().title())


# In[12]:


if __name__ == "__main__":
    main()

