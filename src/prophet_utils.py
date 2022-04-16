# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 21:10:31 2022

@author: 16028
"""

def correctDamageProp(x):
    if x == "K":
        return '1000'
    elif x == "M":
        return '1000000'
    elif x == "B":
        return '1000000'
    elif x == ".K":
        return '1000'
    elif x == ".M":
        return '100000'
    elif x == ".B":
        return '1000000'
    elif x == "K.":
        return '1000'
    elif x == "M.":
        return '100000'
    elif x == "B.":
        return '1000000'
    else:
        return x