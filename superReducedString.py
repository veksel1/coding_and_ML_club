# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 23:29:38 2018

@author: Ilja
"""

#!/bin/python3

import sys

def super_reduced_string(s):
    # Complete this function
    reduction_count=1
    while len(s)>0 and reduction_count>0:
        reduction_count=0
        i = 0
        while i < len(s)-1:
            if s[i]==s[i+1]:
                reduction_count +=1
                s = s[:i] + s[i+2:len(s)]
            i += 1
    
    if s=='':
        return 'Empty String'
    return s 
           
s = input().strip()
result = super_reduced_string(s)
print(result)