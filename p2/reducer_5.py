import sys
import numpy as np

k = 5
# input comes from STDIN
Distances = list()
train_y = list()
for line in sys.stdin:
    distances, y = line.strip().split('\t')
    distances = distances.split(',')
    Distances.append(distances)
    train_y.append(y)

# after appending all the distances, the shape of Distances is #train rows X #test rows
Distances = np.array(Distances, dtype = np.float32).T # taking transpose of Distances, now the shape of Distances is #test rows X #train rows
train_y = np.array(train_y, dtype = np.int8)

# the order of Distances is same as the test data order, eg, row 1 of Distances corresponds to distances of test data row 1 from all the train data
# below 1 row of Distances is the euclidean distances from all the training data
for i, dist in enumerate(Distances):
    k_idx = np.argsort(dist)[:k] # taking the nearest k neighbors
    k_neighbor_distances = list(dist[k_idx].astype(np.unicode_))
    k_neighbor_labels = list(train_y[k_idx].astype(np.unicode_))
    print ('%s\t%s\t%s' % (str(i), ','.join(k_neighbor_distances), ','.join(k_neighbor_labels))) # sending the row#, k nearest neighbors and their corresponding labels to next set of map-reduce job
