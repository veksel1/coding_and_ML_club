# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:27:59 2018

@author: Ilja
"""

#!/bin/python3

import sys

def timeConversion(s):
    # Complete this function
    lenS=len(s)
    
    if s[lenS-2:] == "AM":
        if s[:2] == '12':
            milTime = "00"+s[2:8]
        else:
            milTime = s[:lenS-2]
    else:
        if s[:2] == '12':
            milTime = s
        else:
            milTime = str(int(s[:2])+12)+s[2:8]
    
    return milTime

#s = input().strip()
s = "07:05:45PM"
result = timeConversion(s)
print(result)
