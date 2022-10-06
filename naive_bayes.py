#CSCI 5541 HW4 Naive Baye's Classifier
#Reese Kneeland and Jordyn Ojeda

# Running instructions

# Technical Overview

# Results:

# Your results (precision, recall, and F-Measure)

# Precision
# Recall
# Accuracy
# F-Measure


# How you handled the tokens (i.e. what did you ignore, if anything?)
# We get rid of stop words.

# What smoothing did you use?
# We used add one smoothing


# Did you add any other tricks (i.e. negation-handling, etc.)?


# Is the vocabulary the number of words in the postive plus the number of words in the negative dictionary?
# Is the prior the log(the number of positive documents / total number of documents) = (100 / 200)
# Do we had in add one smoothing for every probability or just when the word doesn't exist
# - log(word fequency plus one / size of the vocabulary)
# add one smoothing is done on the likelihood 
# Is the likelihood log(the positive word frequency in that category / the different types of positve words vocabulary)
# 


from enum import unique
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
        positive_vocab_count = 0
        positive_vocab_unique_count = 0
        negative_vocab_count = 0
        negative_vocab_unique_count = 0
        
        # Create the dictionary of word frequence for the positive movie review data
        for movie_review in range(len(positive_movie_training_data)):
            for word in range(len(positive_movie_training_data[movie_review])):
                
                positive_vocab_count += 1
                
                # If the word is in the dictionary add one to it's count
                if positive_movie_training_data[movie_review][word] in positive_movie_training_dict:
                    positive_movie_training_dict[positive_movie_training_data[movie_review][word]] += 1
                
                # If the key is not in the dictionary add it to the dictionary
                else:
                    positive_movie_training_dict[positive_movie_training_data[movie_review][word]] = 1
                    positive_vocab_unique_count += 1
                     
        # Create the dictionary of word frequence for the positive movie review data
        for movie_review in range(len(negative_movie_training_data)):
            for word in range(len(negative_movie_training_data[movie_review])):
                
                negative_vocab_count += 1
                
                # If the word is in the dictionary add one to it's count
                if negative_movie_training_data[movie_review][word] in negative_movie_training_dict:
                    negative_movie_training_dict[negative_movie_training_data[movie_review][word]] += 1
                
                # If the key is not in the dictionary add it to the dictionary
                else:
                    negative_movie_training_dict[negative_movie_training_data[movie_review][word]] = 1
                    negative_vocab_unique_count += 1
                    
        return positive_movie_training_dict, negative_movie_training_dict, positive_vocab_count, negative_vocab_count, positive_vocab_unique_count, negative_vocab_unique_count
    
    def likelihood(self, word_frequency, vocab_count, unqiue_vocab_words):
        
        likelihood = ((word_frequency + 1) / (vocab_count + unqiue_vocab_words))
        
        return likelihood
    
    def prior(self, number_of_documents, total_documents):
        
        prior = log(number_of_documents / total_documents)
        
        return prior
    
    def predict_stats(self, positive_movie_training_dict, negative_movie_training_dict,
                            positive_movie_dev_set, negative_movie_dev_set,
                            positive_vocab_count, negative_vocab_count,
                            positive_vocab_unique_count, negative_vocab_unique_count):
        
        # Index 0 = True Positive
        # Index 1 = True Negative
        # Index 2 = False Positive
        # Index 3 = False Negative
        confusion_matrix = [0, 0, 0, 0]
        
        for movie_review in range(len(positive_movie_dev_set)):
            positive_prob = 0
            negative_prob = 0
            for word in range(len(positive_movie_dev_set[movie_review])):
                
                negative_frequency = 0
                positive_frequency = 0
                
                current_word = positive_movie_dev_set[movie_review][word]
                
                if(current_word in positive_movie_training_dict):
                    positive_frequency = positive_movie_training_dict[current_word]
            
                    
                if(current_word in negative_movie_training_dict):
                    negative_frequency = negative_movie_training_dict[current_word]
        
                positive_prob += NB.likelihood(positive_frequency, positive_vocab_count, positive_vocab_unique_count) * NB.prior(100, 200)
                negative_prob += NB.likelihood(negative_frequency, negative_vocab_count, negative_vocab_unique_count) * NB.prior(100, 200)
                
            print(positive_prob)  
            if(positive_prob > negative_prob):
                # print(positive_probability)
                # print(negative_probability)
                confusion_matrix[0] += 1
                    
            else:
                confusion_matrix[2] += 1
                

                
        for movie_review in range(len(negative_movie_dev_set)):
            positive_probability = 0
            negative_probability = 0
            for word in range(len(negative_movie_dev_set[movie_review])):
                
                positive_freq = 0
                negative_freq = 0
            
                current_word = negative_movie_dev_set[movie_review][word]
                
                if(current_word in positive_movie_training_dict):
                    positive_freq = positive_movie_training_dict[current_word]
                    
                if(current_word in negative_movie_training_dict):
                    negative_freq = negative_movie_training_dict[current_word]
                
                positive_probability += NB.likelihood(positive_freq, positive_vocab_count, positive_vocab_unique_count) * NB.prior(100, 200)
                negative_probability += NB.likelihood(negative_freq, negative_vocab_count, negative_vocab_unique_count) * NB.prior(100, 200)
                
            if(negative_probability > positive_probability):
                # print(positive_probability)
                # print(negative_probability)
                confusion_matrix[1] += 1
                    
            else:
                confusion_matrix[3] += 1
                    
                    
        return confusion_matrix
                    
                
                
        
                
                
                
    

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
        words_no_punctuation = nltk.tokenize.word_tokenize(data_no_punctuation)
        
        
        # Remove all the stop words that have no meaning
        stop_words = set(stopwords.words('english'))
        
        words = []
        
        for i in range(len(words_no_punctuation)):
            
            if(words_no_punctuation[i] not in stop_words):
                words.append(words_no_punctuation[i])
                
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
        words_no_punctuation = nltk.tokenize.word_tokenize(data_no_punctuation)
        
        
        # Remove all the stop words that have no meaning
        stop_words = set(stopwords.words('english'))
        
        words = []
        
        for i in range(len(words_no_punctuation)):
            
            if(words_no_punctuation[i] not in stop_words):
                words.append(words_no_punctuation[i])
        
        # The data is being split 80 20. 20% in the dev set and 80% in the training set. 
        
        # Every one out of five files will be in the dev set
        if (file % 5 == 0):
            negative_movie_dev_set.append(words)
        else:
            negative_movie_training_data.append(words)
        
        
        
    positive_movie_training_dict, negative_movie_training_dict, positive_vocab_count, neagtive_vocab_count, positive_vocab_unique_count, negative_vocab_unique_count = NB.create_dicts(positive_movie_training_data, negative_movie_training_data)
    
    confusion_matrix = NB.predict_stats(positive_movie_training_dict, negative_movie_training_dict,
                                        positive_movie_dev_set, negative_movie_dev_set,
                                        positive_vocab_count, neagtive_vocab_count,
                                        positive_vocab_unique_count, negative_vocab_unique_count)
    
    # Accuracy = TP + TN / TP + TN + FP + FN
    print(confusion_matrix)
    accuracy = (confusion_matrix[0] + confusion_matrix[1]) / (confusion_matrix[0] + confusion_matrix[1] + confusion_matrix[2] + confusion_matrix[3])
    print(accuracy)
    
    
    



        

