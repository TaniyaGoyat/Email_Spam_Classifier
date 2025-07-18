import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

ps=PorterStemmer()

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")
input_sms=st.text_area("Enter your message")


def transform_text(text):
    # 1
    text = text.lower()

    # 2
    text = word_tokenize(text)

    # 3
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # 4
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # 5
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)



if(st.button('Predict')):
    #Steps to be performed
    # #1 preprocess
    transform_sms=transform_text(input_sms)
    #2 vectorize
    vector_input=tfidf.transform([transform_sms])
    #3 predict
    result=model.predict(vector_input)[0]
    #4 Display
    if result == 1:
        st.header("It's a Spam")
    else:
        st.header("It's a ham! Not Spam")



