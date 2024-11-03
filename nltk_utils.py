import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from nltk.corpus import wordnet
import json
import re

def preprocess_input(sentence):
    replacements = {
        "u": "you",
        "r": "are",
        "thx": "thanks",
        "pls": "please",
        "im": "i am",
        "dont": "do not",
        "cant": "cannot",
        "wont": "will not",
        "ive": "i have",
        "id": "i would",
        "wanna": "want to",
        "gonna": "going to"
    }
    for key, value in replacements.items():
        sentence = re.sub(r'\b{}\b'.format(key), value, sentence)
    return sentence

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag