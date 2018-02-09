# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:05:16 2018

@author: Ilja
"""

def recDigSum(n):
    def tailRecDigSum(n, summa):
        if n // 10 == 0:
            return summa + n % 10
        else:
            summa += n % 10
            return tailRecDigSum(n // 10, summa)

    if n // 10 == 0:
        return n
    else:
        return tailRecDigSum(n, 0)
    

print(recDigSum(111))