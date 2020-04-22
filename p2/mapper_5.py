import numpy as np
import pandas as pd
import sys

#method for normalizing dataset
#start:
def normalize(x):
    x_norm = np.zeros(x.shape)
    mean = np.mean(x, axis = 0)
    sd = np.std(x, axis = 0)
    x_norm = (x - mean) / sd
    return x_norm
#end

test_x = np.array(pd.read_csv('Test.csv'), dtype = np.float32) # reading test data
test_x = normalize(test_x) # normalizing test data

for s in sys.stdin:
    s = s.strip()
    train = np.array(s.split(','), dtype = np.float32)
    x = train[:48]
    y = int(train[48])
    x = np.tile(x, (test_x.shape[0], 1)) # making the train row the size to test_x
    distances = np.linalg.norm(test_x - x, axis = 1) # calculating euclidean distance of all the test data with incoming train row
    distances = list(distances.astype(np.unicode_))
    print ('%s\t%s' % (','.join(distances), y))
