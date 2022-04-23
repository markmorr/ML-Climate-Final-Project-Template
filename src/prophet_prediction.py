# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 19:28:53 2022

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

    
#bucket the damage levels

# https://www.c2es.org/content/tornadoes-and-climate-change/ use this in powerpoint?


# =============================================================================
# directoryPath = r'C:\Users\16028\Downloads\storm_details\details'
# 
# glued_data = pd.DataFrame()
# df_dict = dict()
# 
# res = [f for f in glob.glob(directoryPath+'\*.csv') if "v1.0_d2" in f]# or "123" in f or "a1b" in f]
# i = 0
# for filename in res:
#     i+= 1
#     print(i)
#     x = pd.read_csv(filename)
#     df_dict[i] = x
#     glued_data = pd.concat([glued_data,x],axis=0)
# tempo = glued_data.sample(100000) 
# 
# 

# df1 = glued_data[:1000000]
# 
# df = first_tenth.copy()
# df = glued_data.copy()
# 
# 
# percent_missing = df.isnull().sum() * 100 / len(df)
# print(percent_missing)  
# 
# df.columns = df.columns.str.lower()
# cols_to_use = ['begin_yearmonth', 'begin_day', 'episode_id', 'event_id', 'state', 'event_type', 
#          'magnitude', 'category', 'tor_f_scale', 'tor_length', 'tor_width', 'begin_azimuth',
#          'begin_lat', 'begin_lon', 'damage_property']
# 
# cols_to_drop =['begin_azimuth', 'episode_id', 'category',]
# df = df[cols_to_use]
# df.drop(columns=cols_to_drop, inplace=True)
# 
# df['damage_property'] = df['damage_property'].replace(r'^\s+$', np.nan, regex=True)
# df['damage_property'].fillna('0', inplace=True)  
#  
# yg = df['damage_property'].value_counts()
# print(yg)
# 
# 
# df['damage_property'] = df['damage_property'].apply(correctDamageProp)
# 
# 
# df.damage_property = (df.damage_property.replace(r'[KMB]+$', '', regex=True).astype(float) * \
# df.damage_property.str.extract(r'[\d\.]+([KMB]+)', expand=False)
# .fillna(1).replace(['K','M', 'B'], [10**3, 10**6,10**9 ]).astype(int))
# df['damage_cost'] = df['damage_property']
# 
# # df['damage_cost'].to_csv(interimDataPath + '\saving_damage_corrected.csv', index=False)
# # df.to_csv(interimDataPath + '\interm_dataframe_2000_on.csv', index=False)
# 
# =============================================================================


def getResults(y_true, y_pred):
    print('MAE is: ' + str(mean_absolute_error(y_true,y_pred)))


interimDataPath = r'C:\Users\16028\Downloads\storm_details'
df = pd.read_csv(interimDataPath + '\interm_dataframe_2000_on.csv')

damge_cost_vc = df.damage_cost.value_counts()

df['log_cost'] = np.log(df['damage_cost']+1)

df.log_cost.value_counts()
plt.hist(df['log_cost'])
counts, bins = np.histogram(df.log_cost)
plt.hist(bins[:-1], bins, weights=counts, color='blue')
plt.grid()
plt.show()


df.columns

BASE_YEAR = 1999
df['year'] = df['begin_yearmonth'].astype(str).str[:4].astype(int)
df['month'] = df['begin_yearmonth'].astype(str).str[4:].astype(int)
df['years_since'] = df['year'] - BASE_YEAR

df['days_since'] = df['years_since']*365 + df['month']*12 + df['begin_day']
df.days_since

df.columns

temp = df[:1000]

df.event_type.value_counts()
storm_types = list(df.event_type.value_counts().keys())


storm_exclusion_list = ['Drought', 'Heat', 'Excessive Heat', 'High Surf', 'Wildfire',
                        'Coastal Flood', 'Avalanche', 'Dust Devil', 'Cold/Wind Chill',
                        'Rip Current']

df2 = df.copy()

df = df[~df['event_type'].isin(storm_exclusion_list)]

df['num_storms'] = 1
df.columns

state_vc = df.state.value_counts()

state_exclusion_list = ['ALASKA', 'ATLANTIC NORTH', 'GULF OF MEXICO', 'HAWAII', 'ATLANTIC SOUTH', 'PUERTO RICO', 
 'LAKE MICHIGAN', 'LAKE ERIE', 'LAKE HURON', 'LAKE SUPERIOR',]

def correctDC(x):
    if x == 'DISTRICT OF COLUMBIA':
        return 'VIRGINIA'
    else:
        return x
df['state'] = df['state'].apply(correctDC)
df = df[~df['state'].isin(state_exclusion_list)]
df = df.groupby("state").filter(lambda x: len(x) > 511)


sum_df = df.groupby(['year','month', 'state']).agg({'log_cost': 'sum', 'num_storms': 'sum'})

sum_df = pd.read_csv(r'sum_df_v3.csv')

sum_df3 = df.groupby(['year', 'state']).agg({'log_cost': 'sum', 'num_storms': 'sum'})
sum_df4 = sum_df3.reset_index()
sum_df4 = sum_df4[sum_df4['year']>2010]

sum_df4 = sum_df4.groupby('state').agg({'log_cost': 'sum', 'num_storms':'sum'})


sum_df6 = df.groupby(['state']).agg({'log_cost': 'sum', 'num_storms': 'sum'})
sum_df6.sort_values(by='log_cost', ascending=False, inplace=True)
['Texas', 'Iowa', 'Ohio', 'New York', 'Mississippi', 'Georgia', 'Pennsylvania',
 'Virginia']

# Georgia and Virginia have more storms than mississippi, for instance, 
# Mississippis storms just happen to be more damaing when they do occur

df = sum_df.copy()
df.rename(columns={'log_cost':'y'}, inplace=True)
df.reset_index(inplace=True)

df['day'] = 1
df [['month','day']] =df [['month','day']].astype(str).apply(lambda x: x.str.zfill(2))
df['ds'] = df['year'].astype(str) + '-' + df['month'] +'-' + df['day']


df.y
# sum_df.to_csv('sum_df_v3.csv')
df.columns

# cap the biggest storm values
#find out which states had largest storm values before
# predict new states to have storm values
# rewatch quick segment of stupid video to remind myself



df['y'] = df['y'].clip(upper=1500)

df.y.value_counts()
df.columns

dftest = df[df['state'] == 'ARIZONA']
df.state

df_backup = df.copy()


# future = m.make_future_dataframe(periods=120, freq = 'MS')
# =============================================================================
# for state in df.state.unique():
#     dftest = df[df['state'] == state]
#     # df_inpu
#     m = Prophet(seasonality_mode='multiplicative').fit(dftest[['ds', 'y']])
#     # future = m.make_future_dataframe(periods=3652)
#     print('state')
#     fcst = m.predict(future)
#     fig = m.plot(fcst)
# 
# =============================================================================


reg = RandomForestRegressor()


one_hot_cols = ['state'] #'event_type', 'tor_f_scale']
one_hot = pd.get_dummies(df[one_hot_cols]) 
df = df.drop(one_hot_cols, axis = 1)
df = pd.concat([df, one_hot], axis=1)


df.columns

X = df.drop(columns=['ds', 'y'])
y = df['y']
reg.fit(X,y)

y_pred = reg.predict(X)
y_true = y
mean_absolute_error(y_true, y_pred)


np.exp(32.09)

prec = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\features\pr_climatology_annual-monthly_cmip6_historical_climatology_ensemble_1995-2014_median_USA.csv',
                 skiprows=0, header=1)
prec.rename(columns={'Unnamed: 0': 'state'}, inplace=True)


years = [1980 + i for i in range(0,40)]
months = list(prec.columns)
months.remove("state")
# states = list(df.state) #district of columbia, alaska, hawaii


#creating simulated data for when needed
# =============================================================================
# noise = np.random.normal(value, value/100, size=13)
# my_geo_tuples = []
# for y in years:
#     for m in months:
#         for s in states:
#             value = prec[prec['state'] == s][m]#.values[0]
#             my_geo_tuples.append((y,m,s, value))
# 
# 
# df_sim = pd.DataFrame(data=my_geo_tuples, columns=['year', 'month', 'state', 'precipitation'])
# 
# =============================================================================



#plotting locations
# =============================================================================
# from shapely.geometry import Point
# import geopandas as gpd
# from geopandas import GeoDataFrame
# 
# df = pd.read_csv("Long_Lats.csv", delimiter=',', skiprows=0, low_memory=False)
# 
# df.columns
# geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
# gdf = GeoDataFrame(df, geometry=geometry)   
# 
# #this is a simple map that goes with geopandas
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);
# 
# 
# =============================================================================


col_names = ['ghcn_id', 'lat', 'lon', 'stdv', 'yr_mo', 'network']

df = pd.read_csv('https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-ak-prcp-inv-v1.0.0-20220406',
                 delim_whitespace=True, names= col_names)

df = pd.read_csv('https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-ak-prcp-inv-v1.0.0-20220406',
                 delim_whitespace=True, names= col_names)



col_names = ['id', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug', 'sep',
             'oct', 'nov', 'dec']

df_dict = {}
address_list_dict = {'prec': 'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-pcpnst-v1.0.0-20220406 ',
                'tmax': 'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-tmaxst-v1.0.0-20220406',
                'tmin':'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-tminst-v1.0.0-20220406',
                'avg':'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-tmpcst-v1.0.0-20220406',
                }


for name, address in address_list_dict.items():
    df_dict[name] = pd.read_csv(address,
                      delim_whitespace=True, names= col_names, converters={'id': lambda x: str(x)})
    
    
df_name_list = ['prec', 'tmax', 'tmin', 'avg']

# df = pd.read_csv('https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-pcpnst-v1.0.0-20220406',
#                   delim_whitespace=True, names= col_names, converters={'id': lambda x: str(x)})

# df = df[50000:] #this will get from 1979 on essentially


# =============================================================================
# df = pd.read_csv(r'C:\Users\16028\OneDrive\Documents\prec_example.txt',
#                  delim_whitespace=True, names= col_names, converters={'id': lambda x: str(x)})
# 
# df.columns
#     
# df['state_code'] = df['id'].astype(str).str[:3]
# df['div_number'] = df['id'].astype(str).str[3:4]
# df['element_code'] = df['id'].astype(str).str[4:6]
# df['year'] = df['id'].astype(str).str[6:10]
# 
# 
# number_list = []
# for i in range(1,51):
#     number_list.append(str(i).zfill(3))
# number_list
#     
# 
# df['year'] = df['year'].astype(int)
# df = df[df['year'] > 1970]
# 
# df = df[df['state_code'].isin(number_list)]
# code_to_area_dict = getCodeToAreaDict()
# area_to_code_dict = getAreaToCodeDict()
# df['state'] = df['state_code'].map(code_to_area_dict)
# names_of_month = ['id', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug', 'sep',
#              'oct', 'nov', 'dec']
# df_temp = df[names_of_month]
# 
# df_temp2 = df_temp.melt(id_vars='id')
# month_to_number = monthToNumberDict()
# df_temp2['month'] = df_temp2['variable'].map(month_to_number)
# 
# #######################
# df = df_temp2.copy()
# 
# df['year'] = df['id'].astype(str).str[6:10]
# df['year'] = df['year'].astype(int)
# df['state_code'] = df['id'].astype(str).str[:3]
# df['state'] = df['state_code'].map(code_to_area_dict)
# 
# sum_df = pd.read_csv(r'sum_df_v3.csv')
# sum_df['state'] = sum_df['state'].str.lower()
# 
# df_new = pd.merge(df, sum_df, on =['year', 'month', 'state'], how='inner')
# df_new.to_csv(r'C:\Users\16028\Downloads\storm_figuring_out_stuff\df_new.csv', index=False)
# 
# =============================================================================

#gnn with nodes based off region sharing, neighboring state_ness, climate region sharing?
#some way of encoding all three numbers into one? that would be more like categorical?
#does reinforcement learnign with gpt-3 work?
# also we just need charts and graphs?
#COMMAND TO JUMP TO END OF LINE
# try the gnn with some slightly less engineered features
# gnn compared to rnn?
# because we're predicting in aggregate, we care less about the sequence, is there
# a way to train for that?
# dfdict = pd.DataFrame(columns = ['device_id'])


#closed = left ensures we don't use this month's precipitation average for our value
#3 is the past 3 indices, it relies on perfect regularity, which seems correct at least
# for these states and for this precipitation data





df = pd.read_csv(r'C:\Users\16028\Downloads\storm_figuring_out_stuff\df_new.csv')
df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1)).dt.date
df.set_index('date', inplace=True, drop=True)
# df['moving'] = df.groupby('state')['value'].transform(lambda x: x.rolling(window=k,  closed='left').mean())

feature_name_list = []
window_lengths = [1,3,6,12,24]
feature_list = ['prec']
for k in window_lengths:
    for feature_category in feature_list:
        feature_name = 'past_' + str(k) + '_' + feature_category
        feature_name_list.append(feature_name)
        df[feature_name] = df.groupby('state')['value'].transform(lambda x: x.rolling(window=k,  closed='left').mean())

for feature_name in feature_name_list:
    df[feature_name] = df[feature_name].fillna(method='bfill')

df.reset_index(inplace=True)
base_date = df.date.min()
df['days_since'] = (df['date'] - base_date).dt.days
df['years_since'] = df['year'] - df.year.min()


# =============================================================================
# one_hot_cols = ['state'] #'event_type', 'tor_f_scale']
# one_hot = pd.get_dummies(df[one_hot_cols]) 
# df = df.drop(one_hot_cols, axis = 1)
# df = pd.concat([df, one_hot], axis=1)
# =============================================================================
# =============================================================================
# #RANDOM INTEGER ENCODING
# df['state_encoded'] = df['state'].astype('category').cat.codes
# =============================================================================
###################################################################################
# ALP'S SUGGESTION OF ENCODING BY TARGET ORDERING--HELPED A LOT, .05-.06 increase in R2
df_state_group = df.groupby('state')
te_dict = dict()
for name, group in df_state_group:
    te_dict[name] = group['log_cost'].mean()

te_list_sorted = sorted(te_dict, key=te_dict.get, reverse=True)
te_dict_ranking = {key: rank for rank, key in enumerate(te_list_sorted, 1)}
df['state_encoded'] = df['state'].map(te_dict_ranking)
###################################################################################


cols_to_drop = ['state', 'id', 'variable', 'year', 'date', 'state_code', 'value']
df = df.drop(columns=cols_to_drop)
df.columns

df.columns
df.drop(columns='log_cost')
X = df.drop(columns=['log_cost', 'num_storms']).copy() #'month',
y = df[['log_cost']].copy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size =.50, random_state=9)


def getMetrics(y_true, y_pred, flag ):
    print(flag + ' R2: ' + str(round(r2_score(y_true, y_pred),2)))
    

def runModel(reg, X_train, y_train, X_test, y_test):
    print(reg.__class__.__name__ + '\n\n\n')
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train)
    getMetrics(y_train, y_pred, 'Training')
    y_pred = reg.predict(X_test)
    getMetrics(y_test, y_pred, 'Testing')
    return



# reg = RandomForestRegressor(max_depth=35)
# reg = ExtraTreesRegressor(max_depth=30, min_samples_split=3)

reg_list = [RandomForestRegressor(max_depth=35), ExtraTreesRegressor(max_depth=30, min_samples_split=3),
            GradientBoostingRegressor()]
for reg in reg_list:    
    runModel(reg, X_train, y_train, X_test, y_test)  




from catboost import CatBoostRegressor, Pool

model = CatBoostRegressor(random_seed=9,
                          loss_function='RMSE',
                          logging_level='Info')
model.fit(X_train, y_train)
# Get predictions

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
getMetrics(y_train, y_pred_train, "Training")
getMetrics(y_test, y_pred, "Testing")
runModel(model, X_train, y_train, X_test, y_test)
