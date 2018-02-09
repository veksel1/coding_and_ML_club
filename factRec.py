# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:44:36 2018

@author: Ilja
"""

def recFact(n):
    if n<=1:
        return 1
    else:
        return n*recFact(n-1)
    
print(recFact(5))