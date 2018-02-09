# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:46:50 2018

@author: Ilja
"""

def tailRecFact(n,product=1):
    if n <= 1:
        return product
    else:
        return tailRecFact(n-1,product*n)
        
print(tailRecFact(5))
    