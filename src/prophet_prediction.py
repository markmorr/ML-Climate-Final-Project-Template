# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 19:28:53 2022

@author: 16028
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from prophet_utils import correctDamageProp

if os.getcwd() != r'C:\Users\16028\Downloads\ML-Climate-Final-Project-Template\src':
    os.chdir(r'C:\Users\16028\Downloads\ML-Climate-Final-Project-Template\src')
    print(os.getcwd())
    
# https://www.c2es.org/content/tornadoes-and-climate-change/ use this in powerpoint?

#TODO:
# clean up (use a lambda function) for the dc correction
# filter by storm type
# cap the largest storms
# figure out how prophet takes stuff in
# do a groupby per month and year, per state (average)
# 
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

#want to 

storm_exclusion_list = ['Drought', 'Heat', 'Excessive Heat', 'High Surf', 'Wildfire',
                        'Coastal Flood', 'Avalanche', 'Dust Devil', 'Cold/Wind Chill',
                        'Rip Current']

df2 = df.copy()

df = df[~df['event_type'].isin(storm_exclusion_list)]

# one_hot = pd.get_dummies(df[['event_type', 'tor_f_scale']])
# df = df.drop(['event_type', 'tor_f_scale'], axis = 1)
# df = pd.concat([df, one_hot], axis=1)

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

df_input = sum_df.copy()
df_input.rename(columns={'log_cost':'y'}, inplace=True)
df_input.reset_index(inplace=True)

df_input['day'] = 1
df_input [['month','day']] =df_input [['month','day']].astype(str).apply(lambda x: x.str.zfill(2))
df_input['ds'] = df_input['year'].astype(str) + '-' + df_input['month'] +'-' + df_input['day']


# sum_df.to_csv('sum_df_v3.csv')
df_input.columns

# cap the biggest storm values
#find out which states had largest storm values before
# predict new states to have storm values

from prophet import Prophet
future = m.make_future_dataframe(periods=120, freq = 'MS')


df_input.clip(upper=1500)
df_input.clip(upper=pd.Series({'y': 1500}), axis=1, inplace=True)


def correctY(x):
    if x > 1500:
        return 1500
    return x

df_input['y'] = df_input['y'].apply(correctY)
dftest = df_input[df_input['state'] == '']


for state in df_input.state.unique():
    dftest = df_input[df_input['state'] == state]
    # df_inpu
    m = Prophet(seasonality_mode='multiplicative').fit(dftest[['ds', 'y']])
    # future = m.make_future_dataframe(periods=3652)
    print('state')
    fcst = m.predict(future)
    fig = m.plot(fcst)



