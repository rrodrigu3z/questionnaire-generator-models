"""Utility functions for handling stop words"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words('english'))

def remove_stop_words(sentence):
    """Removes stop words from the specified sentence"""
    word_tokens = word_tokenize(sentence)
    filtered_sentence = []

    for w in word_tokens:
        if w not in STOP_WORDS:
            filtered_sentence.append(w)
    return " ".join(filtered_sentence)
