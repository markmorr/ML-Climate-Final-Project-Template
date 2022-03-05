# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:50:34 2022

@author: 16028
"""

import glob
import pandas as pd

directoryPath = r'C:\Users\16028\Downloads\storm_details'
glued_data = pd.DataFrame()
df_dict = dict()


i = 0
for file_name in glob.glob(directoryPath+'\*.csv'):
    i += 1
    x = pd.read_csv(file_name)
    df_dict[i] = x
    glued_data = pd.concat([glued_data,x],axis=0)
    
print(glued_data.head())
#why are there more columns, what's the mismatch


temp = glued_data.sample(200)


    
    
    