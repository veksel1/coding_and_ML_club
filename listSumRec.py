# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:45:51 2018

@author: Ilja
"""

def listSumRec(aList):
    if type(aList) == int:
        return aList
    elif type(aList) == list:
        if len(aList) == 1:
            return listSumRec(aList[0])
        else:
            return listSumRec(aList[0]) + listSumRec(aList[1:])
        
myList = [1,2,[3,4],[5,6]]
print(listSumRec(myList))