#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
from summarizer import TransformerSummarizer
from newspaper import Article


# In[3]:

@st.cache(allow_output_mutation=True)
def get_model():
    return TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")


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

    if st.button("Summarize"):
        summary = gpt2_model(article,min_length=60)

    st.success(summary)


# In[9]:


if __name__ == "__main__":
    main()

