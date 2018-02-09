# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 23:20:47 2018

@author: Ilja
"""

def homework(n, arr):
    
    print('arr = ', arr)

    cnt = 0
    sorted_arr = sorted(arr)
    print()
    print('sorted_arr = ', sorted_arr)
    print('       arr = ', arr)
#    print('       pos = ', pos)
    print()


    for i in range(len(arr)):
        print('i=',i)
        if arr[i] != sorted_arr[i]:
            print('  arr[i] (before) =', arr[i])
            print('  sorted_arr[i] (before) =', sorted_arr[i])
            cnt += 1
            print('  cnt=',cnt)
            
#            new_idx = pos[sorted_arr[i]]
#            print('  new_idx=',new_idx)
            
#            pos[arr[i]] = new_idx
            for j in range(len(arr)):
                if arr[j]==sorted_arr[i]:
                    break
            arr[i], arr[j] = arr[j], arr[i]
            print('  arr[i] (after) =', arr[i])
#            print('  arr[new_idx] (after) =', arr[new_idx])            
            print('  arr (after) =',arr)
#            print('  pos (after) =',pos)
            

    return cnt


#n = int(input().strip())
#arr = list(map(int, input().str2ip().split()))
arr = [20,50,30,10]
n = len(arr)

asc = homework(n, arr[:])
#desc = homework(n, arr[::-1])
desc = asc
print(min(asc, desc))