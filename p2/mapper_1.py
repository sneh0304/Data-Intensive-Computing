import sys
import string

table = str.maketrans(dict.fromkeys(string.punctuation))
# input comes from STDIN (standard input)
for s in sys.stdin:
    s = s.strip()
    new_s = s.translate(table).lower()
    words = new_s.split()

    for word in words:
        print ('%s\t%s' % (word, 1))
