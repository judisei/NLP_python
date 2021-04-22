# Word2vec mit gensim:

import gensim
import os
#from os import walk
import glob
import time
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

Alle_Dateinamen = glob.glob("data/*.txt")

Korpus = [open(Dateiname, 'r', encoding='utf8').read() for Dateiname in Alle_Dateinamen]



#%% Stop Woerter entfernen
documents = [gensim.utils.simple_preprocess(Text) for Text in Korpus]
stop_Woerter = pd.read_csv('german_stopwords.txt', header=None)[0].values.tolist()

def Text_ohne_Stopwoerter (Text_Liste, stop_Woerter):
    gefilterter_Satz = [w for w in Text_Liste if not w in stop_Woerter]
    return gefilterter_Satz

documents = [Text_ohne_Stopwoerter (Text_Liste, stop_Woerter) for Text_Liste in documents]



#%% Word2Vec Model initialisieren und Vokabular bauen
model = gensim.models.Word2Vec (documents, size=128, window=10, min_count=2, workers=10)

#%% Wort Vektoren trainieren
start = time.time()
model.train(documents, total_examples=len(documents),epochs=100)
end = time.time()
print(end - start) #220sec mit 100epochs, size150, #400 sec size 150, epochs300


#%% Zeige die aehnlichsten Worte
sim= input("Enter term for similarity search: ")
print("\n")
print(model.wv.most_similar (positive=sim))










