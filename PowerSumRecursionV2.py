# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:51:17 2018

@author: Ilja
"""

#import math
#import sys
import math

def powerSumRec(X, N, lowest=1, curIntAmount=1, maxIntAmount=math.inf, inPowSum=0, preLast=0, curLast=1, answer=0):
    print()
    print('input parameters:')
    print('lowest=',lowest,'curIntAmount=',curIntAmount,'maxIntAmount=',maxIntAmount,'inPowSum=',inPowSum,'preLast=',preLast,'curLast=',curLast,'answer=',answer)
    input()
    
    if lowest**N > X:
        print('lowest**N > X')
        print('answer=',answer)
        return answer
    if lowest**N == X:
        return answer + 1
    
    else:
        print('  inPowSum + (curLast)**N=',inPowSum + (curLast)**N)

        if (inPowSum + (curLast)**N >= X):
            print('  inPowSum + (curLast)**N >= X')
            if inPowSum + (curLast)**N == X:
                answer += 1
            maxIntAmount = curIntAmount - 1
            if (curIntAmount > 2):
                return powerSumRec(X, N, lowest, curIntAmount-1, maxIntAmount, inPowSum - preLast**N, max(lowest,preLast-1), preLast+1, answer)
            else:
                return powerSumRec(X, N, lowest=lowest+1, curLast=lowest+1, answer=answer)
        elif (inPowSum + curLast**N < X) and (curIntAmount < maxIntAmount):
            print('  (inPowSum + curLast**N < X) and (curIntAmount < maxIntAmount)')
            return powerSumRec(X, N, lowest, curIntAmount+1, maxIntAmount, inPowSum + curLast**N, curLast, curLast+1, answer)
        elif (inPowSum + curLast**N < X) and (curIntAmount == maxIntAmount):
            print('  (inPowSum + curLast**N < X) and (curIntAmount == maxIntAmount)')
            return powerSumRec(X, N, lowest, curIntAmount, maxIntAmount, inPowSum, preLast, curLast+1, answer)

            
if __name__ == "__main__":
#    X = int(input().strip())
#    N = int(input().strip())
    X=100
    N=2
    result = powerSumRec(X, N)
    print(result)
