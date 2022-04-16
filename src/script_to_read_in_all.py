# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:50:34 2022

@author: 16028
"""

import glob
import pandas as pd
import numpy as np

# from netCDF4 import Dataset as NetCDFFile 
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap

import netCDF4
# nc = netCDF4.Dataset(r'C:\Users\16028\Downloads\climatology-pr-monthly-mean_cmip6_monthly_all-regridded-bct-historical-climatology_median_1995-2014.nc','r')
#the net cdf is clearly 12 months x 

nc = netCDF4.Dataset(r'C:\Users\16028\Downloads\climatology-tasmax-monthly-mean_cmip6_monthly_all-regridded-bct-historical-climatology_median_1995-2014.nc')

print(nc.variables)
ncv = nc.variables
print(ncv.keys())
nc


a = np.zeros(5)

lat = nc.variables['lat'][:]
lat = np.array(lat)


lon = nc.variables['lon'][:]
lon = np.array(lon)



tasmax = nc.variables['climatology-tasmax-monthly-mean'][:]
tasmax = np.array(tasmax)


# 25 to 50 N
# 65 W to 110 W


tasmax_us = tasmax[:,25:50, 65:110]

25*(110-65)

lat_us = lat[25:50]
lon_us = lon[65:110]

tasmax[2][3][9]

# if type(lat) ==  <class 'numpy.ma.core.MaskedArray'>:
lat 
af = np.ma.mean(lat)   

nc.ncinfo()



for k in ncv.keys():
    print(k)
    print(nc.variables[k][:5])
    print('\n')




lat2 = nc.variables['lat']
lon = nc.variables[''][:]
time = nc.variables['time'][:]
t2 = nc.variables['p2t'][:] # 2 meter temperature
mslp = nc.variables['msl'][:] # mean sea level pressure
u = nc.variables['p10u'][:] # 10m u-component of winds
v = nc.variables['p10v'][:] # 10m v-component of winds

directoryPath = r'C:\Users\16028\Downloads\storm_details\details'
glued_data = pd.DataFrame()
df_dict = dict()



i = 0

res = [f for f in glob.glob(directoryPath+'\*.csv') if "v1.0_d2" in f]# or "123" in f or "a1b" in f]
for filename in res:
    i+= 1
    print(i)
    x = pd.read_csv(filename)
    df_dict[i] = x
    glued_data = pd.concat([glued_data,x],axis=0)
    

# https://www.worldclim.org/data/cmip6/cmip6climate.html
    

glued_data.columns
    
tempo = glued_data.sample(10000)    
    

tempo
df = tempo.copy()

percent_missing = df.isnull().sum() * 100 / len(df)
   
    
    
    
    
    
    
# res = [f for f in glob.glob(".txt") if re.match(r'[abc|123|a1b].', f)


print(glued_data.shape)
import os
import re
res = [f for f in os.listdir(path) if re.search(r'(abc|123|a1b).*\.txt$', f)]
res = [f for f in glob.glob(".txt") if re.match(r'[abc|123|a1b].', f)
# for f in res:
    
    

for file_name in glob.glob(directoryPath+'\*.csv'):
    i += 1
    x = pd.read_csv(file_name)
    df_dict[i] = x
    glued_data = pd.concat([glued_data,x],axis=0)
    
print(glued_data.head())
#why are there more columns, what's the mismatch


temp = glued_data.sample(2000)


percent_missing = glued_data.isnull().sum() * 100 / len(df)
df.loc[:, df.isnull().mean() < .30]    

    


