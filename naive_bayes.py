#CSCI 5541 HW4 Naive Baye's Classifier
#Reese Kneeland and Jordyn Ojeda

# Running instructions

# Technical Overview

# Results:

# Your results (precision, recall, and F-Measure)
# How you handled the tokens (i.e. what did you ignore, if anything?)
# What smoothing did you use?
# Did you add any other tricks (i.e. negation-handling, etc.)?


import sys
import random
import os
import nltk


class NaiveBayes:
    def __init__(self):
        # Constants
        self.splitFlag = True

    def validationSplit(self, data):
        
        # tokenize sentences
        the_data = nltk.tokenize.sent_tokenize(data)
        
        # shuffle the dataset for random sampling
        random.shuffle(the_data)
        
        # Use 80% of the dataset
        fileLen = len(the_data)
        split = round(fileLen*0.8)
        
        # 80% of the data
        trainData = the_data[:split]
        
        # remaining 20% of the data
        testData = the_data[split:]

        return trainData, testData

    

if __name__ == '__main__':
    NB = NaiveBayes()
    
    # Positive movie reviews
    postive_movie_reviews_files = os.listdir(sys.argv[1])
    
    # Negative moive reviews
    negative_movie_reviews_files = os.listdir(sys.argv[2])
    
    # Create the positive training data and dev set
    positive_movie_training_data = []
    positive_movie_dev_set = []
    for file in range(len(postive_movie_reviews_files)):
        
        data = open(postive_movie_reviews_files[file], encoding = 'utf8').read()
        
        positive_training_data, positive_dev_set = NB.validationSplit(data)
        
        positive_movie_training_data += positive_training_data
        
        positive_movie_dev_set += positive_dev_set
        
    print(positive_movie_dev_set)
    
    # Create the negative trainging data and dev set
    negative_movie_training_data = []
    negative_movie_dev_set = []
    for file in range(len(negative_movie_reviews_files)):
        
        data = open(negative_movie_reviews_files[file], encoding = 'utf8').read()
        
        negative_training_data, negative_dev_set = NB.validationSplit(data)
        
        negative_movie_training_data += negative_training_data
        
        negative_movie_dev_set += negative_dev_set
        
    print(positive_movie_dev_set)
        

