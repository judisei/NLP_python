
import gensim
import os
#from os import walk
import glob
import time
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import time

n_clusters=2


Alle_Dateinamen = glob.glob("data/*.txt")

Korpus = [open(Dateiname, 'r', encoding='utf8').read() for Dateiname in Alle_Dateinamen]


#%%
from pandas import DataFrame

Korpus_df = DataFrame(Korpus)
print (Korpus_df.head())
print (Korpus_df.shape)
Korpus_df.columns = ['Text']








#%% TFIDF Vektorisierung
stop_words = pd.read_csv('german_stopwords.txt', header=None)[0].values.tolist()
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
#TFIDF_Matrix = tfidf.fit_transform(Korpus_df['Text'])
TFIDF_Matrix = tfidf.fit_transform(Korpus)


#%% Clustering mit NonNegativeMatrix
start = time.time ( )
nmf = NMF ( n_components = n_clusters, random_state = 142, max_iter = 400 )
nmf.fit ( TFIDF_Matrix )
end = time.time ( )
print ( end - start ) # 19 sec

#%%
Anzahl_Worte = 5
for index,topic in enumerate(nmf.components_):
    if (index >-1):
        print(f'Die TOP- #{Anzahl_Worte} Wörter zum Thema #{index}')
        print([tfidf.get_feature_names()[i] for i in topic.argsort()[-Anzahl_Worte:]])
        print('\n')




