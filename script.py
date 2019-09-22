import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')

df = df[['Title','Genre','Director','Actors','Plot']]

#-----KEYWORDS FROM PLOT-----#

# initializing the new column
df['Key_words'] = ""
for index, row in df.iterrows():
    plot = row['Plot']    
    # instantiating Rake, by default it uses english stopwords from NLTK
    # and discards all puntuation characters as well
    r = Rake()
    # extracting the words by passing the text
    r.extract_keywords_from_text(plot)
    # getting the dictionary whith key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()    
    # assigning the key words to the new column for the corresponding movie
    row['Key_words'] = list(key_words_dict_scores.keys())
# dropping the Plot column
df.drop(columns = ['Plot'], inplace = True)

#-----DATA CLEANING-----#

# lowering case
df['Genre'] = df['Genre'].apply(lambda x: x.lower())
df['Director'] = df['Director'].apply(lambda x: x.lower())
df['Actors'] = df['Actors'].apply(lambda x: x.lower())

# merging names 
df['Actors'] = df['Actors'].str.replace(' ', '')
df['Director'] =  df['Director'].str.replace(' ','')

# termination char
df['Genre'] = df['Genre']+','
df['Director'] = df['Director']+','
df['Actors'] = df['Actors']+','

#from list to string
df['Key_words'] = df['Key_words'].apply(lambda x: ','.join(map(str, x)))

# Bag_of_words
df['Bag_of_words'] = df['Genre']+df['Director']+df['Actors']+df['Key_words']
df['Bag_of_words'] = df['Bag_of_words'].str.replace(' ','')
df['Bag_of_words'] = df['Bag_of_words'].str.replace(',',' ')

df.drop(columns = ['Genre','Director','Actors','Key_words'], inplace = True)
print(df.head(100))


#-----MODELING-----#

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['Bag_of_words'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim)

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use in the function to match the indexes
indices = pd.Series(df['Title'])

#  defining the function that takes in movie title 
# as input and returns the top 10 recommended movies
def recommendations(title, cosine_sim = cosine_sim):
    
    # initializing the empty list of recommended movies
    recommended_movies = []
    mov = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append((list(df.index)[i]))
    
    for x in recommended_movies:
        mov.append(df['Title'][x])

    return mov


print(recommendations(title='Pulp Fiction'))
