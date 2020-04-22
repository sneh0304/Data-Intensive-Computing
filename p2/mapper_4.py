import sys

# input comes from STDIN (standard input)
for s in sys.stdin:
    s = s.strip()
    row = s.split('\t', 1)

    print ('%s\t%s' % (row[0], row[1]))
