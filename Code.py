# coding: utf-8

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():

    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()
    #print('download done')


def tokenize_string(my_string):

    return re.findall('[\w\-]+', my_string.lower())

def tokenize(movies):
    """
    movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    movies = tokenize(movies)
    movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    token = []
    for i in movies['genres']:
        token.append(tokenize_string(i))
    movies['tokens'] = token
    return movies
    pass

def train_test_split(ratings):


    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    #print('train ', train, '\n test',test)
    return ratings.iloc[train], ratings.iloc[test]

def cosine_sim(a, b):

    a1 = a.toarray()
    b1 = b.toarray()

    return (np.dot(a1,b1.transpose())[0][0]/(np.linalg.norm(a1)*np.linalg.norm(b1)))
    pass

def mean_absolute_error(predictions, ratings_test):

    return np.abs(predictions - np.array(ratings_test.rating)).mean()

def featurize(movies):
    '''
    movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi|Action'],[789,'Comedy']],columns=['movieId', 'genres'])
    movies = tokenize(movies)
    vocab_keys = sorted(set(sum(movies['tokens'].tolist(), [])))
    print ('keys - ',vocab_keys)
    movies, vocab = featurize(movies)
    print('\nvocab:')
    print(sorted(vocab.items())[:10])
    #Out
    vocab:
    [('action', 0), ('comedy', 1), ('horror', 2), ('romance', 3), ('sci-fi', 4)]

    '''
    # Find unique feature -
    feat = []
    u = set()
    for i in movies.tokens.tolist():
        u.update(i)
    flist = sorted(u)
    # Defining vocab
    x = 0
    vocab = defaultdict(int)
    for i in flist:
        vocab[i]=x
        x+=1
    # Number of documents = rows in movies
    N = movies.shape[0]
    # IDF
    idf = defaultdict(int)
    for f in flist:
        c = 0
        for t in movies.tokens:
            if f in t:
                c+=1
        idf[f]=math.log(N/c)
    # Initialize csr matrix
    mat = []
    # CSR Matrix
    for i,r in movies.iterrows():
        matrow, matcol, matdata = [],[],[]
        tf = Counter()
        tfidf = defaultdict(int)
        mf=0
        tf = Counter(r.tokens)
        mf = max(tf.values())

        for j in r.tokens:
            tfidf[j] = (tf[j]/mf)*(idf[j])
            if j in flist:
                matrow.append(0)
                matcol.append(vocab[j])
                matdata.append(tfidf[j])
        mat = csr_matrix((matdata, (matrow,matcol)), shape=(1,len(vocab)),dtype='float64')
        #movies.set_value(index,'features',csr_matrix((data, (row,col)), shape=(1,len(vocab))))
        feat.extend(mat)
    movies['features'] = feat
    return(movies,vocab)
    pass

def make_predictions(movies, ratings_train, ratings_test):
    '''
    movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi|Action'],[789,'Comedy']], columns=['movieId', 'genres'])
    ratings = pd.DataFrame([[1,123,2],[1,456,4],[1,789,1.5],[2,123,3],[2,456,1],[2,789,3],[3,123,3],[3,456,4],[3,789,2],[4,123,2],[4,456,3],[4,789,2.5],
                            [5,123,2.5],[5,456,4.5],[5,789,5],[6,123,3.5],[6,456,1.5],[6,789,5],[7,123,3.5],[7,456,4.5],[7,789,4],[8,123,2.5],[8,456,3.5],[8,789,4.5],
                            [9,123,4],[9,456,4],[9,789,2.5],[10,123,3],[10,456,2],[10,789,4],[11,123,3],[11,456,4],[11,789,3.5],[12,123,2],[12,456,1],[12,789,3]],
                           columns=['userId','movieId','rating'])
    movies = tokenize(movies)
    vocab_keys = sorted(set(sum(movies['tokens'].tolist(), [])))
    print ('keys - ',vocab_keys)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)

    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)

    #Out
    keys -  ['action', 'comedy', 'horror', 'romance', 'sci-fi']

    vocab:
    [('action', 0), ('comedy', 1), ('horror', 2), ('romance', 3), ('sci-fi', 4)]
    35 training ratings; 1 testing ratings
    [ 2.75]
    '''

    # M = movie, U = user, P = predict
    Prate = []
    for i in ratings_test.itertuples():
        cos = 0
        rate = 0
        rateTrain = 0
        v = 0
        for j in ratings_train.itertuples():
            if i[1]==j[1]:
                x = movies.loc[movies['movieId']==i[2],'features']
                y = movies.loc[movies['movieId']==j[2],'features']
                x_pred = x.iloc[0]
                y_pred = y.iloc[0]
                cosval = cosine_sim(x_pred,y_pred) #cosine similarity
                rate+=cosval*j[3]
                cos+=cosval
                rateTrain+=j[3]
                v+=1
        if cos>0 and rate>0:
            r = rate/cos
        else:
            r = rateTrain/v
        Prate.append(r)
    #print('Prate - ',Prate)
    return np.array(Prate)
    pass

def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    vocab_keys = sorted(set(sum(movies['tokens'].tolist(), [])))
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
