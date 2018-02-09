# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 13:16:51 2018

@author: ilja.surikovs
"""

#!/bin/python3

#import sys

def lilysHomework(arr):
    print(arr)
    asc_arr = arr.copy()
    desc_arr = arr.copy()
    print()
    asc_arr.sort()
    print(asc_arr)
    desc_arr.sort(reverse=True)
    print(desc_arr)
    asc_swap_count = 0
    desc_swap_count = 0
    # Complete this function
    for i in range(len(arr)):
        print(i)
        if arr[i] != asc_arr[i]:
            asc_swap_count += 1
        if arr[i] != desc_arr[i]:
            desc_swap_count += 1
    min_swap_count = min(asc_swap_count, desc_swap_count)
    if min_swap_count % 2 == 1:
        swap_count = (min_swap_count // 2) + 1
    else:
        swap_count = swap_count // 2
    
    return swap_count
   
if __name__ == "__main__":
    #n = int(input().strip())
    #arr = list(map(int, input().strip().split(' ')))
    arr=[2,5,3,1]
    result = lilysHomework(arr)
    print(result)
