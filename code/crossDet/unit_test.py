# from utils import *

# poly = [1,2]
# px,py=1,4
# print(np.polyval(poly,px))
# res = point_distance_line(px,py,poly)
# print(res)


import time
start =time.clock()
sum=0
for i in range(1,101):
    sum=sum+i 
    print(sum )
end = time.clock()
print('Running time: %s Seconds'%(end-start))