import sys
from collections import defaultdict

current_empID = None
empID = None

# input comes from STDIN
for line in sys.stdin:
    line = line.strip()

    empID, rest = line.split('\t', 1)
    restList = rest.split('\t')
    if current_empID != empID:
        if current_empID:
            print ('%s\t%s' % (current_empID, _dict[current_empID]))
        _dict = defaultdict(list)
        current_empID = empID

# below code makes sure that name should be 1st column after employeeID
    if len(restList) == 1:
        _dict[empID].insert(0, restList[0])
    else:
        _dict[empID].extend(restList)


if current_empID == empID:
    print ('%s\t%s' % (current_empID, _dict[current_empID]))
