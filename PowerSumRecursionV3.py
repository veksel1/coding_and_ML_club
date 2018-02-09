# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:51:17 2018

@author: Ilja
"""

#import sys

def powerSum(X, N):
    highest = int(X**(1/N))+1
    
    def rec(X=X, N=N, summa=0, num=1, highest=highest):
        if summa > X:
            return 0
        elif summa == X:
            return 1
        elif num > highest:
            return 0
        else:
            return rec(X, N, summa+num**N, num+1) + rec(X, N, summa, num+1)
    
    return rec()
    
if __name__ == "__main__":
#    X = int(input().strip())
#    N = int(input().strip())
    X=100
    N=2
    result = powerSum(X, N)
    print(result)
