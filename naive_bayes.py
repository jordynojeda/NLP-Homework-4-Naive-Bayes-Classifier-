#CSCI 5541 HW4 Naive Baye's Classifier
#Reese Kneeland and Jordyn Ojeda

# Running instructions

# To run the program type: "python naive_bayes.py"
# The program assumes that there is a folder named "movie_reviews" in the same directory as the script 
# naive_bayes.py. There should also be two folders inside "movie_reviews" folder "pos" and folder "neg"
# whichs store the positive and negative movie reviews

# Technical Overview


# Data seperation 
# The data was seperated 80% in training data and 20% in dev data. One out of every five senetneces 
# was put into the dev set and the other four data sets were put into the training data. 

# Results:

# Your results (Accuray, Precision, Recall, and F-Measure)

# Accuracy:    84.00%
# Precision:   79.50%
# Recall:      87.36%
# F-Measure:   83.25%


# How you handled the tokens (i.e. what did you ignore, if anything?)
# We get rid of stop words, as they don't correlate to sentiment and would skew our probabilities.

# What smoothing did you use?
# We used add one smoothing Laplace smoothing to account for zero word probabilities, and built a
# dictionary of word probabilities in each set to compute the frequency. The frequency is then aggregated
# at the prediction step along with the class priors to compute a classification.


# Did you add any other tricks (i.e. negation-handling, etc.)?
# We got rid of punctuation to make sure our words were tokenized correctly, and not create duplicate entries.



import sys
import os
import re
import math
import nltk
from nltk.corpus import stopwords


class NaiveBayes:
    def __init__(self):
        # Constants
        self.splitFlag = True
    
    
    # Create the positive and negative movie review dictionaries with word frequencies
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
    
    
    # Function to calculate the likelihood
    def likelihood(self, word_frequency, vocab_count, unqiue_vocab_words):
        
        likelihood = ((word_frequency + 1) / (vocab_count + unqiue_vocab_words))
        
        return math.log(likelihood)
    
    # Function to calculate the prior
    def prior(self, number_of_documents, total_documents):
        
        prior = number_of_documents / total_documents
        
        return math.log(prior)
    
    
    # Predict if it's a positive or negative moview review and store it in a 
    # confusion matrix
    def predict_stats(self, positive_movie_training_dict, negative_movie_training_dict,
                            positive_movie_dev_set, negative_movie_dev_set,
                            positive_vocab_count, negative_vocab_count,
                            positive_vocab_unique_count, negative_vocab_unique_count):
        
        # Index 0 = True Positive
        # Index 1 = True Negative
        # Index 2 = False Positive
        # Index 3 = False Negative
        confusion_matrix = [0, 0, 0, 0]
        
        positive_dev_set_length = len(positive_movie_dev_set)
        negative_dev_set_length = len(negative_movie_dev_set)
        combined_dev_set_length = (positive_dev_set_length + negative_dev_set_length)
        
        for movie_review in range(len(positive_movie_dev_set)):
            positive_prob = 0
            negative_prob = 0
            for word in range(len(positive_movie_dev_set[movie_review])):
                
                negative_frequency = 0
                positive_frequency = 0
                
                # Get the current word of the negative moview review
                current_word = positive_movie_dev_set[movie_review][word]
                
                # Get the words frequency if it's in the dictionary otherwise keep the frequency at 0
                if(current_word in positive_movie_training_dict):
                    positive_frequency = positive_movie_training_dict[current_word]
            
                # Get the words frequency if it's in the dictionary otherwise keep the frequency at 0
                if(current_word in negative_movie_training_dict):
                    negative_frequency = negative_movie_training_dict[current_word]

                # Calculate the probabilities 
                positive_prob += NB.likelihood(positive_frequency, positive_vocab_count, positive_vocab_unique_count) + NB.prior(positive_dev_set_length, combined_dev_set_length)
                negative_prob += NB.likelihood(negative_frequency, negative_vocab_count, negative_vocab_unique_count) + NB.prior(negative_dev_set_length, combined_dev_set_length)
                
            # Compare the probabilities and store them in the confusion matrix
            if(positive_prob > negative_prob):
                confusion_matrix[0] += 1
                    
            else:
                confusion_matrix[2] += 1
                

                
        for movie_review in range(len(negative_movie_dev_set)):
            positive_probability = 0
            negative_probability = 0
            
            for word in range(len(negative_movie_dev_set[movie_review])):
                
                positive_freq = 0
                negative_freq = 0
            
                # Get the current word of the negative moview review
                current_word = negative_movie_dev_set[movie_review][word]
                
                # Get the words frequency if it's in the dictionary otherwise keep the frequency at 0
                if(current_word in positive_movie_training_dict):
                    positive_freq = positive_movie_training_dict[current_word]
                    
                # Get the words frequency if it's in the dictionary otherwise keep the frequency at 0
                if(current_word in negative_movie_training_dict):
                    negative_freq = negative_movie_training_dict[current_word]
                
                # Calculate the probabilities 
                positive_probability += NB.likelihood(positive_freq, positive_vocab_count, positive_vocab_unique_count) + NB.prior(positive_dev_set_length, combined_dev_set_length)
                negative_probability += NB.likelihood(negative_freq, negative_vocab_count, negative_vocab_unique_count) + NB.prior(negative_dev_set_length, combined_dev_set_length)
                
            # Compare the probabilities and store them in the confusion matrix
            if(negative_probability > positive_probability):
                confusion_matrix[1] += 1
                    
            else:
                confusion_matrix[3] += 1
                    
                    
        return confusion_matrix
                    
                

if __name__ == '__main__':
    NB = NaiveBayes()
    
    positive_movie_training_dict = {}
    negative_movie_training_dict = {}
    
    # Positive movie reviews
    postive_movie_reviews_files = os.listdir(os.getcwd() + "/movie_reviews/pos")
    
    # Negative moive reviews
    negative_movie_reviews_files = os.listdir(os.getcwd() + "/movie_reviews/neg")
    
    # Create the positive training data and dev set
    positive_movie_training_data = []
    positive_movie_dev_set = []
                
    for file in range(len(postive_movie_reviews_files)):
        
        # Open the movie review
        data = open(os.path.join(os.getcwd(), "movie_reviews/pos/" + str(postive_movie_reviews_files[file])), encoding = 'utf8').read()
        
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
        data = open(os.path.join(os.getcwd(), "movie_reviews/neg/" + str(negative_movie_reviews_files[file])), encoding = 'utf8').read()
        
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
    
    
    # Index 0 = True Positive
    # Index 1 = True Negative
    # Index 2 = False Positive
    # Index 3 = False Negative
    confusion_matrix = NB.predict_stats(positive_movie_training_dict, negative_movie_training_dict,
                                        positive_movie_dev_set, negative_movie_dev_set,
                                        positive_vocab_count, neagtive_vocab_count,
                                        positive_vocab_unique_count, negative_vocab_unique_count)
    
    # Accuracy = TP + TN / TP + TN + FP + FN
    accuracy = (((confusion_matrix[0] + confusion_matrix[1]) / (confusion_matrix[0] + confusion_matrix[1] + confusion_matrix[2] + confusion_matrix[3])) * 100)
    
    # Precision = TP / TP + FP
    precision = ((confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[2])) * 100)
    
    # Recall = TP / TP + FN
    recall = ((confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[3])) * 100)
    
    # F-Meausre = TP / TP + 0.5(FP + FN)
    f_measure = ((confusion_matrix[0] / (confusion_matrix[0] + (0.5 * (confusion_matrix[2] + confusion_matrix[3])))) * 100)
    
    print("Statistical Results:")
    print( f"Accuracy:    {accuracy:.2f}%")
    print( f"Precision:   {precision:.2f}%")
    print( f"Recall:      {recall:.2f}%")
    print( f"F-Measure:   {f_measure:.2f}%")
    
    
    



        

