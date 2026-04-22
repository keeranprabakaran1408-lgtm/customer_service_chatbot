import nltk#imports all the nltk packages to be used
import numpy as np
#nltk.download('punkt') #This is used to download the training library of nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()#to stem words to core

class Chatbot():#class
    def tokenize(sentence):#function for tokenization of words/sentence for input
            return nltk.word_tokenize(sentence)

    def stem(token_word):#function to stem words to core
            return stemmer.stem(token_word)

    def bag_of_words(tokenized_sentence, token_word):  # function to stem and tokenize words
        tokenized_sentence = [Chatbot.stem(i) for i in tokenized_sentence]

        bag = np.zeros(len(token_word), dtype=np.float32)  # set each token to zero and iterate for an index
        for idx, i in enumerate(token_word):  # index and current position of character in token
            if i in tokenized_sentence:
                bag[idx] = 1
        return bag

Chatbot()




