import sys
import string
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer

keyWords = ['science', 'sea' , 'fire']
table = str.maketrans(dict.fromkeys(string.punctuation)) # a mapping table for removing punctuations from string
# input comes from STDIN (standard input)
for s in sys.stdin:
    s = s.strip()
    new_s = s.translate(table).lower() # removing punctuations from string and then converting them to lowercase

    tokens = word_tokenize(new_s) # converting the line into tokens of words
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens] # lemmatizing the tokens
    trigrams = ngrams(tokens, 3) # creatinf 3-grams

    for trigram in trigrams:
        if keyWords[0] in trigram:
            temp = ['$' if x == keyWords[0] else x for x in trigram]
            print ('%s\t%s' % ('_'.join(temp), 1))
        elif keyWords[1] in trigram:
            temp = ['$' if x == keyWords[1] else x for x in trigram]
            print ('%s\t%s' % ('_'.join(temp), 1))
        elif keyWords[2] in trigram:
            temp = ['$' if x == keyWords[2] else x for x in trigram]
            print ('%s\t%s' % ('_'.join(temp), 1))
