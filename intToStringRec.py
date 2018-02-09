# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:36:45 2018

@author: Ilja
"""

def intToString(num):
    if num // 10 == 0:
        return str(num)
    else:
        return intToString(num // 10) + str(num % 10)

num = 12345 
print(intToString(num))