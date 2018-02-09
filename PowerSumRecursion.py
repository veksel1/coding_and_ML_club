# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:51:17 2018

@author: Ilja
"""

#import math
#import sys

def powerSumRec(X, N, intAmount=1, powSum=0, preLastInt=0, answer=0):
    print()
    print('input parameters:')
    print('X=',X,'N=',N,'intAmount=',intAmount,'powSum=',powSum,'prelastInt=',preLastInt,'answer=',answer)
#    print('lastIntPowered=',lastIntPowered)
    print('powSum + (prelastInt+1)**N=',powSum + (preLastInt+1)**N)
    if powSum + (preLastInt+1)**N>= X:
        print('powSum + (prelastInt+1)**N >= X')
        if powSum == X:
            answer += 1
        if intAmount > 1:
            powerSumRec(X, N, intAmount-1, powSum - preLastInt**N, preLastInt+1, answer)
        else:
            return answer
    else:
        print('powSum + (prelastInt+1)**N < X')
        powerSumRec(X, N, intAmount+1, powSum, preLastInt+1, answer)
        
if __name__ == "__main__":
#    X = int(input().strip())
#    N = int(input().strip())
    X=10
    N=2
    result = powerSumRec(X, N)
    print(result)
