import bz2
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]

with bz2.open(in_file, "rt") as fi:
    with open(out_file, 'w') as fo:
        count = 0
        head = True
        for line in fi:
            arr = line.rstrip('\n').split('\t')
            if len(arr) == 2:
                if head:
                    head = False
                    continue
                fo.write("%s,%s,%d\n" % (arr[0], arr[1], count))
                count += 1
                if count > 1000000:
                    break
