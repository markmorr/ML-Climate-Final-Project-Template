# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:32:11 2022

@author: 16028
"""

from ftplib import FTP
import os


# citing this page for instructions on fetching data: https://nsidc.org/support/64231694-FTP-Client-Data-Access#ftp
destdir = r"C:\Users\16028\Downloads\rutgers_snow"
password = "markmorr@usc.edu"
# directory = '/DATASETS/NOAA/G02135/north/daily/data'

# directory = 'https://nsidc.org/data/G10035/versions/1'
# directory = 'https://nsidc.org/data/SNEX20_UNM_GPR/versions/1'
# directory = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G10035'
directory = 'DATASETS/NOAA/G10035'
############################################
### Don't need to change this code below ###
############################################
# FTP server
ftpdir = 'sidads.colorado.edu'

#Connect and log in to the FTP
print('Logging in')
ftp = FTP(ftpdir)
ftp.login('anonymous',password)

# Change to the directory where the files are on the FTP
print('Changing to '+ directory)
ftp.cwd(directory)

# Get a list of the files in the FTP directory
files = ftp.nlst()
files = files[2:]
print(files)

#Change to the destination directory on own computer where you want to save the files
os.chdir(destdir)

#Download all the files within the FTP directory
for file in files:
    print('Downloading...' + file)
    ftp.retrbinary('RETR ' + file, open(file, 'wb').write)

#Close the FTP connection
ftp.quit()



import pandas as pd
data = pd.read_csv(r'C:\Users\16028\Downloads\rutgers_snow\N_seaice_extent_climatology_1981-2010_v3.0.csv',
                   skiprows=1)


data_rutgers = pd.read_csv(r'C:\Users\16028\Downloads\rutgers_snow\G10035-rutgers-nh-24km-weekly-sce-v01r00-19800826-20210906.nc')

# conda install -c anaconda xarray
import xarray as xr
import netCDF4
ds = xr.open_dataset(r'C:\Users\16028\Downloads\rutgers_snow\G10035-rutgers-nh-24km-weekly-sce-v01r00-19800826-20210906.nc')

df = ds.to_dataframe()
