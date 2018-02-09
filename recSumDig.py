# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:05:16 2018

@author: Ilja
"""

def recDigSum(n):
    if n // 10 == 0:
        return n
    else:
        return n % 10 + recDigSum(n // 10)

print(recDigSum(45))