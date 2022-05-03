# -*- coding: utf-8 -*-
"""
Created on Mon May  2 01:14:48 2022

@author: 16028
"""


import os
if os.getcwd() != r'C:\Users\16028\Downloads\ML-Climate-Final-Project-Template\src':
    os.chdir(r'C:\Users\16028\Downloads\ML-Climate-Final-Project-Template\src')
    print(os.getcwd())
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import datetime as dt
from datetime import timedelta
import glob

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


from prophet import Prophet



from temp_state_dict_working import getCodeToAreaDict, getAreaToCodeDict, monthToNumberDict
from prophet_utils import correctDamageProp
from state_to_neighbor_dict import get_state_border_dict

# TODOs  
# bucket the damage levels
# change it to two largest nearby states neighbors?
# setup sklearn custom pipeline to perform a search over different state neighbor dictionaries
# okay so the date sorting is screwed up

# https://www.c2es.org/content/tornadoes-and-climate-change/ use this in powerpoint?

# =============================================================================
directoryPath = r'C:\Users\16028\Downloads\storm_details\details'

glued_data = pd.DataFrame()
df_dict = dict()

res = [f for f in glob.glob(directoryPath+'\*.csv') if "v1.0_d2" in f]# or "123" in f or "a1b" in f]
i = 0
for filename in res:
    i+= 1
    print(i)
    x = pd.read_csv(filename)
    df_dict[i] = x
    glued_data = pd.concat([glued_data,x],axis=0)

df = glued_data.copy()
percent_missing = df.isnull().sum() * 100 / len(df)
print(percent_missing)  


df.columns = df.columns.str.lower()
cols_to_use = ['begin_yearmonth', 'begin_day', 'episode_id', 'event_id', 'state', 'event_type', 
          'magnitude', 'category', 'tor_f_scale', 'tor_length', 'tor_width', 'begin_azimuth',
          'begin_lat', 'begin_lon', 'damage_property']

cols_to_drop =['begin_azimuth', 'episode_id', 'category',]
df = df[cols_to_use]
df.drop(columns=cols_to_drop, inplace=True)

df['damage_property'] = df['damage_property'].replace(r'^\s+$', np.nan, regex=True)
df['damage_property'].fillna('0', inplace=True)  
df['damage_property'] = df['damage_property'].apply(correctDamageProp)
df.damage_property = (df.damage_property.replace(r'[KMB]+$', '', regex=True).astype(float) * \
df.damage_property.str.extract(r'[\d\.]+([KMB]+)', expand=False)
.fillna(1).replace(['K','M', 'B'], [10**3, 10**6,10**9 ]).astype(int))
df['damage_cost'] = df['damage_property']

df['damage_cost'].to_csv(interimDataPath + '\saving_damage_corrected.csv', index=False)
df.to_csv(interimDataPath + '\interm_dataframe_2000_on.csv', index=False)
# =============================================================================

# interimDataPath = r'C:\Users\16028\Downloads\storm_details'
# df = pd.read_csv(interimDataPath + '\interm_dataframe_2000_on.csv')































