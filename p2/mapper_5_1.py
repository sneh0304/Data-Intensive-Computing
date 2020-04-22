import sys

# input comes from STDIN (standard input)
for s in sys.stdin:
    s = s.strip()

    print ('%s\t%s' % (1, s)) # adding key as 1 for all the imputs, so that all the output goes to same reducer
