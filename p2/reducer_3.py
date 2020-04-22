import sys
from collections import defaultdict

current_word = None
word = None

# input comes from STDIN
for line in sys.stdin:
    line = line.strip()

    word, docID = line.split('\t')

    if current_word != word:
        if current_word:
            print ('%s\t%s' % (current_word, sorted(wordDict[current_word])))
        wordDict = defaultdict(set) # using set as we don't need duplicate docIDs
        current_word = word

    wordDict[word].add(docID)

if current_word == word:
    print ('%s\t%s' % (current_word, sorted(wordDict[current_word])))
