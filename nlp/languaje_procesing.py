
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'',text)
          

def romove_url(text):     
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'',text)

def remove_numbers(text):
    numbers = re.compile('\d+')
    return numbers.sub('',text)    

def remove_unnecesary_punctuation(text):
    unnecesary_punctuations = re.compile('[^\w\s]')
    return unnecesary_punctuations.sub(r'',text)    

def normalize_text(text): 
    """ """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    punctuation = string.punctuation 
    normalized_text = []
    text = remove_html(text)
    text = remove_numbers(text)
    text = remove_unnecesary_punctuation(text) #si se demoran mucho los algoritmos usar una estructura de datos
    tokenize_text = word_tokenize(text)
    for token in tokenize_text:
        token = token.lower()
        token = lemmatizer.lemmatize(token)
        if token not in stop_words and token not in punctuation:
            normalized_text.append(token)

    return normalized_text


