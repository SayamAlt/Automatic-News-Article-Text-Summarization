# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
from newspaper import Article

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))


@st.cache(allow_output_mutation=True)
def getModel():
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def scrapArticle(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        article.nlp()
    except Exception as e:
        print(e)
    return article.text

def generateSummary(text,tokenizer,model):
    input_ids = tokenizer(
        [WHITESPACE_HANDLER(text)],
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]
    
    output_ids = model.generate(
            input_ids=input_ids,
            max_length=150,
            no_repeat_ngram_size=2,
            num_beams=5
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
                                                                                                                    
    return summary
    
def main():
    st.title("Automatic News Articles Text Summarization")
    link = st.text_input("Enter the URL of news channel:")
    article = scrapArticle(link)
    summary = ""
    tokenizer, model = getModel();
    
    if st.button("Summarize"):
        summary = generateSummary(article,tokenizer,model);

    st.success(summary)

if __name__ == "__main__":
    main()
