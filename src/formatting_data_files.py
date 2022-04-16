# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 09:59:16 2022

@author: 16028
"""

import pandas as pd
import numpy as np

#use daily average to simulate data, upward adjusted climate data

PATH = 'C:\\Users\\16028\\Downloads\\storm_details\\features\\'
glob.glob(PATH)



# https://www.ncdc.noaa.gov/stormevents/faq.jsp
i = 0

res = [f for f in glob.glob(directoryPath+'\*.csv') if "v1.0_d2" in f]# or "123" in f or "a1b" in f]
for filename in res:
    i+= 1
    print(i)
    x = pd.read_csv(filename)
    df_dict[i] = x
    glued_data = pd.concat([glued_data,x],axis=0)


for f in glob.glob(PATH):
    print(f)
df = pd.read_csv(r'C:\Users\16028\Downloads\pr_timeseries_annual_cmip6_ssp126_ensemble_2015-2100_medianUSA.csv')









#questions--do you think my noise thing is a reasonable thing to do? in a way I hoped
# that it would help with uncertainty

#high