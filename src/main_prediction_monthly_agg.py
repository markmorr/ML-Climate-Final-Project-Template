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
from useful_code import correctDamageProp
from state_to_neighbor_dict import get_state_border_dict



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






# runModel(model, X_train, y_train, X_test, y_test)
# Get predictions

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# y_pred_train = model.predict(X_train)
# getMetrics(y_train, y_pred_train, "Training")
# getMetrics(y_test, y_pred, "Testing")


########################################################################################
#######################################################################################
### check whether catboost performs just as well leaving state as categorical (maybe it does
# that stuff internally?)

#insert the backfill before I feed it into the dictionary?
# clean this code up

#insert the backfill in the rolling_feature_dict for loop?
# start passing in feature_tuple_list as a parameter to ensure uniformity 
# ensure that I'm testing on temporally future data 
#(do a time series rolling split? figure out how to do that?)
        
# in order to not deal with missing features(i.e. uniform length), need to get average of neighbors?
# past 24_prec seems to be having some issues?
# add past_36 and maybe even past_48 ?
# try passing in raw stuff to an MLP
# split final report into sections: stuff that worked, stuff that did not work but was
# still interesting, and stuff that did not work but I included to show the process I took
# to get here
# post on 4771 edstem?






def getRollingFeatures(df_input, feature_tuple_list, feature_name_list):
    df_input['date'] = pd.to_datetime(df_input[['year', 'month']].assign(DAY=1)).dt.date
    df_input.sort_values(by=['state', 'date'], inplace=True)
    df_input.set_index('date', inplace=True, drop=True)
    # df_input['moving'] = df_input.groupby('state')['value'].transform(lambda x: x.rolling(window=k,  closed='left').mean())
    
    # feature_name_list = []
    # window_lengths = [1,3,6,12,24]
    # feature_list = ['prec']
    for feat_tuple, feat_name, in zip(feature_tuple_list, feature_name_list):
        # feature_name = 'past_' + str(k) + '_' + feature_category
        # feature_name_list.append(feature_name)
        # feature_name_list.append(feat)
        # SOON VALUE WILL HAVE TO BE REPLACED BY THE CHARACTERISTIC FEATURE
        k = feat_tuple[1]
        print(k)
        print(feat_name)
        df_input[feat_name] = df_input.groupby('state')['value'].transform(lambda x: x.rolling(window=k,  closed='left').mean())

    # df_input.groupby('state')['value'].transform(lambda x: x.rolling(window=k,  closed='left').mean())
    
    for feature_name in feature_name_list:
        df_input[feature_name] = df_input[feature_name].fillna(method='bfill')
    
    df_input.reset_index(inplace=True)
    base_date = df_input.date.min()
    return df_input

window_lengths = [1,3,6,12,24]
feature_list = ['prec']
feature_name_list = []
feature_tuple_list = []
for feat_name in feature_list:
    for period in window_lengths:
        feature_tuple_list.append((feat_name, period))
        feature_name_list.append( feat_name + '_' + str(period))


df_to_use = pd.read_csv(r'C:\Users\16028\Downloads\storm_figuring_out_stuff\df_new.csv')
getRollingFeatures(df_to_use, feature_tuple_list, feature_name_list)


#####################################################################################

df = pd.read_csv(r'C:\Users\16028\Downloads\storm_figuring_out_stuff\df_new.csv')
df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1)).dt.date
df.sort_values(by=['state', 'date'], inplace=True)
df.set_index('date', inplace=True, drop=True)
state_list = list(df['state'].unique())


state_to_neighbor_dict = get_state_border_dict()
# remove keys that aren't included
# iterate through the state and it's neighbor list--> if a state in the neighbor list
# is unlisted, then remove that one from the list (but keep the rest of course)     
remove = [k for k in state_to_neighbor_dict if k not in state_list]
for k in remove: 
    del state_to_neighbor_dict[k]
for k,v in state_to_neighbor_dict.items():
    if k not in state_list:        
        print(k)
    for state_name in v:
        if state_name not in state_list:
            v.remove(state_name) 
        if state_name == k:
            v.remove(state_name)
            
df['state_neighbors'] = df['state'].map(state_to_neighbor_dict)


df
rolling_feature_dict = dict()
k_list = [1,3,6,12,24]
feature_category = 'prec'
for state in state_list:
    for k in k_list:
        rolling_feature_dict['prec_' + state + '_' + str(k)] = df[df['state'] == state]['value'].transform(lambda x: x.rolling(window=k,  closed='left').mean())


feature_tuple_list
feature_name_list
rolling_feature_dict = dict()
for state in state_list:
    for feat_tuple, feat_name in zip(feature_tuple_list, feature_name_list):
        print(feat_tuple)
        k = feat_tuple[1]
        rolling_feature_dict[feat_name + '_' + state] = df[df['state'] == state]['value'].transform(lambda x: x.rolling(window=k,  closed='left').mean())

df.reset_index(inplace=True)


def getNeighboringFeatures(row, k):
    n_neighbors = len(row['state_neighbors'])
    date = row['date']
    contribution = 0
    for neighbor in row['state_neighbors']:

        neighbor_series = rolling_feature_dict['prec_' + str(k) + '_' + neighbor]
            # print(neighbor_series)
        if date in neighbor_series.index:
            contribution += neighbor_series[date]
        else:
            n_neighbors -= 1
    if n_neighbors <= 0:
        return 0
    else:
        return contribution/n_neighbors
    
added_feature_list = []
for k in [1,3,6,12,24]:
    feature_name = 'neighbor_prec_' + str(k)
    added_feature_list.append(feature_name)
    df[feature_name] = df.apply(getNeighboringFeatures, args =(k,), axis=1)
    df[feature_name] = df[feature_name].fillna(method='bfill')

    
added_feature_list = []
for feat_tuple, feat_name in zip(feature_name_list, feature_tuple_list):
    feature_name = 'neighbor_prec_' + str(k)
    added_feature_list.append(feat_name)
    df[feature_name] = df.apply(getNeighboringFeatures, args =(k,), axis=1)
    df[feature_name] = df[feature_name].fillna(method='bfill')

# in a way the yearly precipitation is probably useful as well
# try it without the engineered features, just datetime, to see how it does?
df_with_neighbors = df.copy()
# df_new = pd.merge(df_to_use, df, on =['year', 'month', 'state'], how='inner')


###############################################################################
###############################################################################



df = pd.read_csv(r'C:\Users\16028\Downloads\storm_figuring_out_stuff\df_new.csv')
df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1)).dt.date
df.set_index('date', inplace=True, drop=True)
# df['moving'] = df.groupby('state')['value'].transform(lambda x: x.rolling(window=k,  closed='left').mean())


### this is the feature engineering for the state's own variables
 
window_lengths = [1,3,6,12,24] #***
feature_list = ['prec']
feature_name_list = []
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
# #RANDOM INTEGER ENCODING
# df['state_encoded'] = df['state'].astype('category').cat.codes
# =============================================================================
# =============================================================================
###############################################################################
# TARGET ENCODING
# ALP'S SUGGESTION OF ENCODING BY TARGET ORDERING--HELPED A LOT, .05-.06 increase in R2
df_state_group = df.groupby('state')
te_dict = dict()
for name, group in df_state_group:
    te_dict[name] = group['log_cost'].mean()

te_list_sorted = sorted(te_dict, key=te_dict.get, reverse=True)
te_dict_ranking = {key: rank for rank, key in enumerate(te_list_sorted, 1)}
df['state_encoded'] = df['state'].map(te_dict_ranking)
# =============================================================================
###############################################################################
key_columns = ['year', 'month', 'state']
df = pd.merge(df, df_with_neighbors[['year', 'month', 'state'] + added_feature_list], on=key_columns, how='inner')


cols_to_drop = ['state', 'id', 'variable', 'year', 'date', 'state_code', 'value']
df = df.drop(columns=cols_to_drop)
df.drop(columns='log_cost')
X = df.drop(columns=['log_cost', 'num_storms']).copy() #'month',
y = df[['log_cost']].copy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size =.50, random_state=9)


def getMetrics(y_true, y_pred, flag ):
    print(flag + ' R2: ' + str(round(r2_score(y_true, y_pred),2)))
    

def runModel(reg, X_train, y_train, X_test, y_test):
    print(reg.__class__.__name__)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train)
    getMetrics(y_train, y_pred, 'Training')
    y_pred = reg.predict(X_test)
    getMetrics(y_test, y_pred, 'Testing')
    print('\n\n\n')
    return


from catboost import CatBoostRegressor, Pool
model = CatBoostRegressor(random_seed=9,
                          loss_function='RMSE',
                          logging_level='Silent')


reg_list = [RandomForestRegressor(max_depth=35), ExtraTreesRegressor(max_depth=30, min_samples_split=3),
            GradientBoostingRegressor(),
            CatBoostRegressor(random_seed=9, loss_function='RMSE',logging_level='Silent')]
for reg in reg_list:    
    runModel(reg, X_train, y_train, X_test, y_test)  




# build up the code
# # then move on to climate forward 
# then expand the feature set
# # binning/discretizing response storm amounts
# # then go lower level to zip code
# # # then connect it to housing prices
# might need to do some work to ensure the neighboring variables always match with
# the state's own features, but that may not be the case--> maybe just rearrange code

# fix the index reset issue
# pass in feature list as a parameter
# get df.apply (args = (,)) to work




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
    
    

df_dict = dict()
directory_path = r"C:/Users/16028/Downloads/storm_figuring_out_stuff/local_noaa"
res = [f for f in glob.glob( directory_path + r"/" +'*.csv')]# or "123" in f or "a1b" in f]
feature_category_list = ['prec', 'tmin', 'tmpc', 'tmax']
# ' \ ' and ' / ' are interchangeable in python paths
for filename in res:

    print(filename)
    file_id = filename.split('raw_')[1]
    file_id = file_id.split('.csv')[0]
    print()
    print(file_id)
    if file_id in feature_category_list:
        x = pd.read_csv(filename, converters={'id': lambda x: str(x)}) # delim_whitespace=True,
        x.reset_index(drop=True)
        df_dict[file_id] = x



df_dict.keys()
df_dict['tmpc']
#

