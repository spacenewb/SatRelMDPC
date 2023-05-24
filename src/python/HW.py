import sys
import numpy as np

n_args = len(sys.argv) - 1
print("Passed Arguments: " + str(n_args))

arglist = []

n = 1
for i in sys.argv[1:]:
    curr_arg = i
    interim_list = str(curr_arg).strip('][').split(' ')
    interim_list[:] = [float(x) for x in interim_list if x]
    arglist.append(np.asarray(interim_list))
    print("Arg " + str(n) + ": " + str(arglist[n-1]))
    n = n+1