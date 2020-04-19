import sys
import string

table = str.maketrans(dict.fromkeys(string.punctuation)) # a mapping table for removing punctuations from string
# input comes from STDIN (standard input)
for s in sys.stdin:
    s = s.strip()

    new_s = s.translate(table).lower() # removing punctuations from string and then converting them to lowercase
    words = new_s.split()

    for word in words:
        print ('%s\t%s' % (word, 1))
