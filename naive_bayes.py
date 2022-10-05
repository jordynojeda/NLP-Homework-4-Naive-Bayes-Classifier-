#CSCI 5541 HW4 Naive Baye's Classifier
#Reese Kneeland and Jordyn Ojeda

# Running instructions

# Technical Overview

# Results:

# Your results (precision, recall, and F-Measure)
# How you handled the tokens (i.e. what did you ignore, if anything?)
# What smoothing did you use?
# Did you add any other tricks (i.e. negation-handling, etc.)?

# Is the vocabulary the number of words in the postive plus the number of words in the negative dictionary?
# Is the prior the log(the different types of postitve words / the total number of words)
# Do we had in add one smoothing for every probability or just when the word doesn't exist
# Is the likelhood log(the word probability / the vocabulary)
# 


import sys
import random
import os
import re
import math
from math import log
import nltk
from nltk.corpus import stopwords


class NaiveBayes:
    def __init__(self):
        # Constants
        self.splitFlag = True
    
    def create_dicts(self, positive_movie_training_data, negative_movie_training_data):
        
        positive_movie_training_dict = {}
        negative_movie_training_dict = {}
        
        # Create the dictionary of word frequence for the positive movie review data
        for movie_review in range(len(positive_movie_training_data)):
            for word in range(len(positive_movie_training_data[movie_review])):
                
                # If the word is in the dictionary add one to it's count
                if positive_movie_training_data[movie_review][word] in positive_movie_training_dict:
                    positive_movie_training_dict[positive_movie_training_data[movie_review][word]] += 1
                
                # If the key is not in the dictionary add it to the dictionary
                else:
                    positive_movie_training_dict[positive_movie_training_data[movie_review][word]] = 1
                     
        # Create the dictionary of word frequence for the positive movie review data
        for movie_review in range(len(negative_movie_training_data)):
            for word in range(len(negative_movie_training_data[movie_review])):
                
                # If the word is in the dictionary add one to it's count
                if negative_movie_training_data[movie_review][word] in negative_movie_training_dict:
                    negative_movie_training_dict[negative_movie_training_data[movie_review][word]] += 1
                
                # If the key is not in the dictionary add it to the dictionary
                else:
                    negative_movie_training_dict[negative_movie_training_data[movie_review][word]] = 1
                    
        return positive_movie_training_dict, negative_movie_training_dict
                
                
                
    

    # def laplace_smoothing(self, n_label_items, vocab, word_counts, word, text_label):
    #     a = word_counts[text_label][word] + 1
    #     b = n_label_items[text_label] + len(vocab)
    #     return math.log(a/b)

    # def group_by_label(x, y, labels):
    #     data = {}
    #     for l in labels:
    #         data[l] = x[np.where(y == l)]
    #     return data

    # def fit(self, x, y, labels):
    #     n_label_items = {}
    #     log_label_priors = {}
    #     n = len(x)
    #     grouped_data = group_by_label(x, y, labels)
    #     for l, data in grouped_data.items():
    #         n_label_items[l] = len(data)
    #         log_label_priors[l] = math.log(n_label_items[l] / n)
    #     return n_label_items, log_label_priors

if __name__ == '__main__':
    NB = NaiveBayes()
    
    positive_movie_training_dict = {}
    negative_movie_training_dict = {}
    
    # Positive movie reviews
    postive_movie_reviews_files = os.listdir(sys.argv[1])
    
    # Negative moive reviews
    negative_movie_reviews_files = os.listdir(sys.argv[2])
    
    # Create the positive training data and dev set
    positive_movie_training_data = []
    positive_movie_dev_set = []
                
    for file in range(len(postive_movie_reviews_files)):
        
        # Open the movie review
        data = open(os.path.join(os.getcwd(), "pos/" + str(postive_movie_reviews_files[file])), encoding = 'utf8').read()
        
        # Remove the punctuation from the file
        data_no_punctuation = re.sub(r'[^a-zA-Z\s]', "", data)
        
        # Make the file into words
        words = nltk.tokenize.word_tokenize(data_no_punctuation)
        
        # The data is being split 80 20. 20% in the dev set and 80% in the training set. 
        
        # Every one out of five files will be in the dev set
        if (file % 5 == 0):
            positive_movie_dev_set.append(words)
        else:
            positive_movie_training_data.append(words)
            
    
    # Create the negative trainging data and dev set
    negative_movie_training_data = []
    negative_movie_dev_set = []
    for file in range(len(negative_movie_reviews_files)):
        
        # Open the movie review
        data = open(os.path.join(os.getcwd(), "neg/" + str(negative_movie_reviews_files[file])), encoding = 'utf8').read()
        
        # Remove the punctuation from the file
        data_no_punctuation = re.sub(r'[^a-zA-Z\s]', "", data)
        
        # Make the file into words
        words = nltk.tokenize.word_tokenize(data_no_punctuation)
        
        # The data is being split 80 20. 20% in the dev set and 80% in the training set. 
        
        # Every one out of five files will be in the dev set
        if (file % 5 == 0):
            negative_movie_dev_set.append(words)
        else:
            negative_movie_training_data.append(words)
        
        
        
    positive_movie_training_dict, negative_movie_training_dict = NB.create_dicts(positive_movie_training_data, negative_movie_training_data)



        

