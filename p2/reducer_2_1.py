import sys
from collections import defaultdict

wordDict = dict()

# input comes from STDIN
for line in sys.stdin:
    line = line.strip()

    temp, ngram, count = line.split('\t')

    try:
        count = int(count)
    except ValueError:
        continue

    wordDict[ngram] = count

sortedByValue = sorted(wordDict.items(), key = lambda x : x[1], reverse = True) # sorting the trigrams as per their counts in descending order

# printing the top 10 trigrams with max counts
for k, v in sortedByValue[: 10]:
    print ('%s\t%s' % (k, v))
