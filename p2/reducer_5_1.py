import sys
import numpy as np
from collections import defaultdict

k = 5
Distances = defaultdict(list)
Labels = defaultdict(list)
# input comes from STDIN
for line in sys.stdin:
    line = line.strip()
    temp, i, k_neighbor_distances, k_neighbor_labels = line.split('\t')
    k_neighbor_distances = k_neighbor_distances.split(',')
    k_neighbor_labels = k_neighbor_labels.split(',')
    Distances[int(i)].extend(k_neighbor_distances) # appending all the distances together coming from different reducers from job1
    Labels[int(i)].extend(k_neighbor_labels) # appending all the labels together coming from different reducers from job1

for key, val in sorted(Distances.items()):
    val = np.array(val, dtype = np.float32)
    k_idx = np.argsort(val)[:k] # getting the global k nearest neighbors
    final_k_neighbor_labels = np.array(Labels[key], dtype = np.int8)[k_idx] # getting the labels corresponding to k_idx
    most_common, count = np.unique(final_k_neighbor_labels, return_counts = True) # finding all the unique labels and their counts
    res = most_common[np.argmax(count)] # the label with the max count is the predicted label for this test row
    print ('%s\t%s' % (key, res))
