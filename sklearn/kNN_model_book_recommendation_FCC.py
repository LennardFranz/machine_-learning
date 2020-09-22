


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


books_filename = 'book-crossings/BX-Books.csv'
ratings_filename = 'book-crossings/BX-Book-Ratings.csv'

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

#print(df_books)


df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# add your code here - consider creating a new cell for each section of code

df = df_ratings

user_200 = df['user'].value_counts()
books_100 = df['isbn'].value_counts()

df = df[df['user'].isin(user_200[user_200 > 200].index)]
df = df[df['isbn'].isin(books_100[books_100 > 100].index)]

df = pd.merge(right=df, left=df_books, on='isbn')

df = df.drop_duplicates(['title', 'user'])
piv = df.pivot(index='title', columns='user', values='rating').fillna(0) 



model_NN = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors= 6)

model_NN.fit(piv)

distance, indices = model_NN.kneighbors(piv, n_neighbors=6)

def get_recommends(book = ""):
  list_recommandations = [book,[]]
  isbn_book = piv.index.get_loc(str(book))
  for i in range(5,0,-1):
    list_kNN = [piv.index[indices[isbn_book][i]], distance[isbn_book][i]]
    list_recommandations[1].append(list_kNN) 

  recommended_books = list_recommandations
  return recommended_books


#books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
#books = get_recommends('The Queen of the Damned (Vampire Chronicles (Paperback))')

#print(books)

import random

def random_book_recommend():
    random_index = random.randint(0, piv.index.shape[0])

    book = piv.index[random_index]
    return get_recommends(book = book)

#books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
#books = get_recommends('The Queen of the Damned (Vampire Chronicles (Paperback))')
#examples

#print(books)
#print title of example and nearest neigbors with distance

print(random_book_recommend())
#print random book with nearest neigbors with distance
