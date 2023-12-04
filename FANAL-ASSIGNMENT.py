#!/usr/bin/env python
# coding: utf-8

# Marwah jamaah AL-johany
# 
# 4150893
# 
# F5
# 
# ASSIGNMENT : Information Visualization in Information  Retrieval 

# ------------------------------------------------------------------------------------------------

# # 1-apply the Soundex algorithm 

# In[1]:


import re
from collections import defaultdict


# In[3]:


import re
def get_soundex_code(word):
    #Convert the word to uppercase
    word = word.upper()
    #Remove non-alphabetic characters
    word = re.sub(r'[^A-Z]', '', word)
    if not word:
        return ''
    soundex_code = word[0]
    #encoding map is a dictionary that maps groups of letters to their corresponding Soundex digits. 
    encoding_map = {'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3', 'L': '4', 'MN': '5', 'R': '6'}
    #Calculate the Soundex code
    for encoding, digit in encoding_map.items():
        soundex_code += ''.join(digit if letter in encoding else '' for letter in word[1:])
     #Remove consecutive duplicate digits   
    soundex_code = re.sub(r'(.)\1+', r'\1', soundex_code)
    #Remove the first character
    soundex_code = soundex_code.replace(soundex_code[0], '', 1)
    #Remove vowels
    soundex_code = re.sub(r'[AEIOUY]', '', soundex_code)
    #Pad the Soundex code
    soundex_code = soundex_code.ljust(4, '0')
    soundex_code = soundex_code[:4]
    #Return the Soundex code
    return soundex_code


# In[5]:


def build_soundex_index(words):
#This object will store the Soundex codes as keys and the corresponding words as values. 
    soundex_index = defaultdict(list)
#Iterate over each word in the input list   
    for word in words:
#Calculate the Soundex code for the word        
        soundex_code = get_soundex_code(word)
#Add the word to the Soundex index    
        soundex_index[soundex_code].append(word)
#Return the Soundex index    
    return soundex_index
    
                


# In[6]:


#This function allows you to find words that sound similar to the query word based on their Soundex codes.
def find_similar_sounding_words(query, soundex_index):
#Calculate the Soundex code for the query word        
    soundex_code = get_soundex_code(query)
#Return the list of similar sounding words    
    return soundex_index[soundex_code]


# In[58]:


words = ['Marwah', 'Marwan', 'Marwen', 'Marwa', 'and','the','or','hi','Marwan', 'Marwin', 'Marwen', 'Marwah', 'Marwaan', 'Marwane', 'Marwah', 'Marwah', 'Marwa', 'Marwan', 'Marwen', 'Marwan', 'Marwen', 'Marwah', 'Marwaan', 'Marwane', 'Marwah', 'Marwah', 'Marwa', 'Marwan', 'Marwen', 'Marwan', 'Marwen', 'Marwah', 'Marwaan', 'Marwane', 'Marwah', 'Marwah', 'Marwa', 'Marwan', 'Marwen', 'Marwan', 'Marwen', 'Marwah', 'Marwaan', 'Marwane', 'Marwah','Marwina', 'Marwinaa', 'Marwinaan', 'Marwanah', 'Marwanaa', 'Marwaneh', 'Marwaneey', 'Marwenah', 'Marwee', 'Marwee', 'Marwee', 'Marween', 'Marweena', 'Marweenah', 'Marwahh', 'Marwahhh', 'Marwahhhh', 'Marwahhhhh', 'Marwahhhhhh', 'Marwahhhhhhh']
soundex_index = build_soundex_index(words)
query = 'Marwah'
similar_words = find_similar_sounding_words(query, soundex_index)
print(f"Words similar to '{query}':")
print ('____________________________________________________\n')
print(similar_words)
print(soundex_index.items())
# Save the similar words in a variable
similar_words_output = similar_words


# # 2-generates a word cloud for soundex output (English text) 
# 

# In[20]:


pip install wordcloud


# In[62]:


# Import the necessary libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Sample data
Data = similar_words_output
# Convert the list of words into a string
data_string = ' '.join(Data)
# Create word cloud
Cloud_Setting = WordCloud(background_color='white',
                          stopwords=['and', 'the', 'is', 'they', 'in', 'of', 'or', 'a', 'are']).generate(data_string)

# Display the generated word cloud
plt.imshow(Cloud_Setting)
# Hide the axes and ticks
plt.axis('off')
# Show the plot
plt.show()


# # ------------------------------------------------------------------------------------------------

# # Another example : apply the  Boolean Retrieval Model
# 

# In[3]:


# importing the necessary libraries and downloading the required resources from the NLTK library:
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')


# In[4]:


#preprocess_document :takes a document as input and performs a series of text preprocessing steps on it
def preprocess_document(document):
#uses the word_tokenize function from the NLTK library to split the document into individual words or tokens.
#The resulting tokens are stored in the tokens variable.
    tokens = word_tokenize(document)
# 1-converts all the tokens to lowercase using a list comprehension.
#2-filters out any tokens that are not entirely composed of alphabetic characters
    tokens = [token.lower() for token in tokens if token.isalpha()]
#retrieves a set of English stopwords from the NLTK library.
    stop_words = set(stopwords.words('english'))
#filters out any tokens that are present in the set of stopwords. The resulting tokens are stored back in the tokens variable.    
    tokens = [token for token in tokens if token not in stop_words]
#creates an instance of the PorterStemmer class from the NLTK library.The PorterStemmer is a widely used stemming algorithm.
    stemmer = PorterStemmer()
# applies stemming to each token in the tokens list using the stemmer created in the previous step.    
    tokens = [stemmer.stem(token) for token in tokens]
#returns the tokens list, which represents the preprocessed version of the input document.    
    return tokens


# In[6]:


document = "Heatmaps use color gradients to represent the density or intensity of data values across \n"            "a two-dimensional space. They are commonly used in IR to visualize patterns and distributions. \n"            "For example, in web analytics, heatmaps can show the areas of a webpage that receive the most \n"            "user attention or engagement, helping to optimize website design and user experience. In \n"            "information retrieval systems, heatmaps can also be used to analyze user interactions, such as \n"            "click patterns or dwell time, to improve search result relevance or user interface design."

preprocessed_tokens = preprocess_document(document)
print(preprocessed_tokens)


# # generates a word cloud for Boolean Retrieval Model output (English text)

# In[25]:


# Import the necessary libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Sample data
Data = preprocessed_tokens
# Convert the list of words into a string
data_string = ' '.join(Data)
# Create word cloud
Cloud_Setting = WordCloud(background_color='white',
                          stopwords=['and', 'the', 'is', 'they', 'in', 'of', 'or', 'a', 'are']).generate(data_string)

# Display the generated word cloud
plt.imshow(Cloud_Setting)
# Hide the axes and ticks
plt.axis('off')
# Show the plot
plt.show()


# # generates a word cloud for Arabic text (Optional step)

# In[45]:


#import the necessary libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display

# Arabic text data
Arabic_text = "كيف تستطيع أمّة أن تصنع مستقبلها وهي في قبضة ماضيها يعصرها عصراً حتى يستنفذ كل ذرة من طاقاتها ؟"
# Reshape and reorder Arabic text
reshaped_text = arabic_reshaper.reshape(Arabic_text)
text_display = get_display(reshaped_text)

# Create word cloud
wordCloud = WordCloud(font_path='C:/Users/marwa/Downloads/alfont_com_VEXA-thin.ttf',
                      background_color='white').generate(text_display)

# Display the generated word cloud
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')

# Show the plot
plt.show()


# # ------------------------------------------------------------------------------------------------

# # Apply intersecting Two Postings Lists and Display  result in a scatter plot

# In[74]:


#intersecting two lists ,iterating through each list and comparing the elements. The resulting intersection a new list 
def intersect_postings_list(postings_list1, postings_list2):
    #Initialize an empty list or set to store the intersected elements.
    intersected_list = []
    i, j = 0, 0
    #Iterate through each element in the first list.
    #For each element, check if it also exists in the second list.
    #If the element is found in the second list, add it to the intersected list or set.\
    #Once all elements in the first list have been processed, the resulting list or set will contain the intersection of the two lists.
    while i < len(postings_list1) and j < len(postings_list2):
        if postings_list1[i] == postings_list2[j]:
            intersected_list.append(postings_list1[i])
            i += 1
            j += 1
        elif postings_list1[i] < postings_list2[j]:
            intersected_list.append(postings_list1[i])
            i += 1
        else:
            intersected_list.append(postings_list2[j])
            j += 1
# Append the remaining elements, if any
    while i < len(postings_list1):
        intersected_list.append(postings_list1[i])
        i += 1
    while j < len(postings_list2):
        intersected_list.append(postings_list2[j])
        j += 1
    return intersected_list


# In[75]:


import matplotlib.pyplot as plt

# Example lists
list1 = [1, 2, 3, 4, 5,9,10,11,12,17,18,19]
list2 = [4, 5, 6, 7, 8,22,14,15,16,20,24,27,30]

# Intersection of the lists
intersected_list = intersect_postings_list(list1, list2)

# Display the intersected_list in a scatter plot
x = intersected_list  # X-axis values
y = [0] * len(intersected_list)  # Y-axis values (zeros for this example)

plt.scatter(x, y)
plt.xlabel('Intersected Elements')
plt.ylabel('Y')
plt.title('Scatter Plot of Intersected Elements')
plt.show()


# # ------------------------------------------------------------------------------------------------

# # generates a scatter plot using the matplotlib.pyplot library  and random data (Optional step)

# In[63]:


#import the necessary libraries
#matplotlib.pyplot as plt for creating plots, and numpy as np for generating random data.
import matplotlib.pyplot as plt
import numpy as np
#ensures that the random numbers generated for the x, y ,colors
#and sizes arrays will be the same each time the code is executed 
np.random.seed(0)
#generates an array of 400 random numbers from a standard normal distribution for both x and y coordinates. 
x = np.random.randn(400)
y = np.random.randn(400)
colors = np.random.rand(400)
#generate an array of random integers (lower ,upper,size )
sizes = np.random.randint(10, 40, 400)

# Create scatter plot
plt.scatter(x, y, c=colors, s=sizes, alpha=0.80, cmap='PiYG')

# Customize plot appearance
plt.title("random Scatter Plot visualization techniques")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
# Show the plot
plt.show()


# # ------------------------------------------------------------------------------------------------
