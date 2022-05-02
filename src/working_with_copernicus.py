# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:30:01 2022

@author: 16028
"""

import pandas as pd

import netCDF4
nc = netCDF4.Dataset(r'C:\Users\16028\Downloads\copernicus_temp\tas_temp.nc', 'r')

# file2read = netCDF4.Dataset(cwd+'\filename.nc','r')
var1 = nc.variables['tas']  # access a variable in the file

b_dict = nc.variables
nc
b_dict.keys()
print(5)

from scipy.io import netcdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print(b_dict)


import xarray as xr


file2read = netCDF4.Dataset(r'C:\Users\16028\Downloads\copernicus_temp\tas_temp.nc', 'r')
temp = file2read.variables['tas'] # var can be 'Theta', 'S', 'V', 'U' etc..
data = temp[:]*1
file2read.close()

file2read.variables
print(file2read['tas'].shape)

time_array = nc['time_bnds'][:]
time_array
1850*30


735110/365

30/365



import os
os.getcwd()
import cdsapi
files = [f for f in os.listdir('.') if os.path.isfile(f)]


c = cdsapi.Client()

myvar = 'precipitation'
var_list = ['near_surface_air_temperature', 'precipitation']
c.retrieve(
    'projections-cmip6',
    {
        'format': 'zip',
        'temporal_resolution': 'monthly',
        'experiment': 'ssp2_4_5',
        'level': 'single_levels',
        'variable': myvar,
        'model': 'cesm2',
        'date': '2000-01-01/2051-01-01',
        'area': [
            55, -130, 20,
            -60,
        ],
    },
    r'C:\Users\16028\Downloads\copernicus_' + myvar + '.zip')


file2read = netCDF4.Dataset(r'C:\Users\16028\Downloads\copernicus_precipitation\precipitation_cesm2.nc', 'r')
nc = netCDF4.Dataset(r'C:\Users\16028\Downloads\copernicus_precipitation\precipitation_cesm2.nc', 'r')


print(file2read.variables.keys())
print(nc.variables.keys())

file2read['pr']
df_dict = dict()
for var_name in nc.variables.keys():
    df_dict[var_name] = nc[var_name][:].data
    
time_prec = nc['pr'][:].data
time_prec.shape
37*57

my_new_array = time_prec.data

lat_array_cop = nc['lat'][:].data
lon_array_cop = nc['lon'][:].data
df_lat = pd.DataFrame(nc['lat'][:].data ,columns=['lat'])
df_lon = pd.DataFrame(nc['lon'][:].data, columns=['lon'])

new_array = lat_array_cop

arr1 = np.array([1,2])
arr2 = np.array([3,4,5])
arr3 = np.array(arr1, arr2)
lat_array_cop
output_arr = np.zeros((lat_array_cop.shape[0], lon_array_cop.shape[0]))
for x in lat_array_cop:
    for y in lon_array_cop:
        output_arr[i,j] = [x,y]
        
df_lat_bns = pd.DataFrame(nc['lat_bnds'][:].data, columns=['lat_bnds1', 'lat_bnds2'])
df_lon_bns = pd.DataFrame(nc['lon_bnds'][:].data, columns=['lon_bnds1', 'lon_bnds2'])

df8= pd.DataFrame(nc['lat'][:], columns=['lat'])

df8= pd.DataFrame(my_new_array, columns=['t1', 't2'])
748250/365
import numpy as np
import numpy.ma as ma

df_lon*(-1) + 180
df_lon_bns

df['t1'] = df['t1'] /365
df['t2'] = df['t2'] /365


lat

36*12

#precipitation feels awful small
# units conversion??
df_dict['lat']
df_dict['lon']
#1.250 lon
#.942
a = np.array(df_dict['lat'])
a.diff()
np.ediff1d(a)

37*57

import plotly.express as px
import pandas as pd

df = pd.read_csv("location_coordinate.csv")

fig = px.scatter_geo(df,lat='lat',lon='long', hover_name="id")
fig.update_layout(title = 'World map', title_x=0.5)
fig.show()

import geopandas as gpd

shapefile = gpd.read_file(r"C:\Users\16028\Downloads\CONUS_CLIMATE_DIVISIONS.shp\GIS.OFFICIAL_CLIM_DIVISIONS.shp")
gdf = gpd.read_file(r"C:\Users\16028\Downloads\CONUS_CLIMATE_DIVISIONS.shp\GIS.OFFICIAL_CLIM_DIVISIONS.shp")

print(shapefile)
shapefile
shapefile.head()
fig, ax = plt.subplots(figsize = (15,15))
shapefile.plot(ax=ax)

shapefile.columns
shapefile['NAME'].value_counts()
shapefile.value_counts()


af = shapefile.geometry.centroid
# DF = 
#compute the difference in accuracy using the CMIP6 on 2014 data vs. the actual on 
# 2014 data?

from shapely.geometry import Point, Polygon
shapefile['ST_ABBRV'].value_counts()


fig, ax = plt.subplots(figsize = (15,15))
af.plot(ax=ax)

af.plot()
#
gdf["x"] = gdf.centroid.x
gdf["y"] = gdf.centroid.y
gdf["x"]
gdf["centroids"] = gdf.centroid
gdf["centroids"]
