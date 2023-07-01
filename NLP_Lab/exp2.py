# Perform lemmatization and stemming using python library nltk.


import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer=WordNetLemmatizer()
print(lemmatizer.lemmatize("running"))
print(lemmatizer.lemmatize("runs"))
def lemmatize(word):
    lemmatizer=WordNetLemmatizer()
    print("Verb Form: "+lemmatizer.lemmatize(word,pos="v"))
    print("Noun Form: "+lemmatizer.lemmatize(word,pos="n"))
    print("Adverb Form: "+lemmatizer.lemmatize(word,pos="r"))
    print("Adjective Form: "+lemmatizer.lemmatize(word,pos="a"))
lemmatize('skewing')



'''
Output:

running
run
Verb Form: skew
Noun Form: skewing
Adverb Form: skewing
Adjective Form: skewing

'''


import nltk
from nltk.stem import PorterStemmer, LancasterStemmer
porter_stemmer=PorterStemmer()
print(porter_stemmer.stem('running'))
print(porter_stemmer.stem('runs'))
print(porter_stemmer.stem('ran'))

'''

Output:

run
run
ran
'''