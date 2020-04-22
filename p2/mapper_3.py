import sys
import string
import os

table = str.maketrans(dict.fromkeys(string.punctuation)) # a mapping table for removing punctuations from string
# input comes from STDIN (standard input)
for s in sys.stdin:
    s = s.strip()
    new_s = s.translate(table).lower() # removing punctuations from string and then converting them to lowercase
    words = new_s.split()
    docPath = os.getenv('map_input_file') # getting the map_input_file path
    docID = 0
    # assigning docIDs 
    if 'arthur.txt' in docPath:
        docID = 1
    elif 'james.txt' in docPath:
        docID = 2
    elif 'leonardo.txt' in docPath:
        docID = 3

    for word in words:
        print ('%s\t%s' % (word, docID))
