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


from sklearn.ensemble import GradientBoostingRegressor
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


from temp_state_dict_working import getCodeToAreaDict, getAreaToCodeDict, monthToNumberDict
from prophet_utils import correctDamageProp

if os.getcwd() != r'C:\Users\16028\Downloads\ML-Climate-Final-Project-Template\src':
    os.chdir(r'C:\Users\16028\Downloads\ML-Climate-Final-Project-Template\src')
    print(os.getcwd())
    
    
    
#bucket the damage levels
# try various one hot encoding levels
#rereate sum df but by year now??
    
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

#want to 

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
states = list(df.state) #district of columbia, alaska, hawaii
states

# noise = np.random.normal(value, value/100, size=13)
my_geo_tuples = []
for y in years:
    for m in months:
        for s in states:
            value = prec[prec['state'] == s][m]#.values[0]
            my_geo_tuples.append((y,m,s, value))


df_sim = pd.DataFrame(data=my_geo_tuples, columns=['year', 'month', 'state', 'precipitation'])







from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

# df = pd.read_csv("Long_Lats.csv", delimiter=',', skiprows=0, low_memory=False)

df.columns
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf = GeoDataFrame(df, geometry=geometry)   

#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);




col_names = ['ghcn_id', 'lat', 'lon', 'stdv', 'yr_mo', 'network']

df = pd.read_csv('https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-ak-prcp-inv-v1.0.0-20220406',
                 delim_whitespace=True, names= col_names)

df = pd.read_csv('https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-ak-prcp-inv-v1.0.0-20220406',
                 delim_whitespace=True, names= col_names)



col_names = ['id', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug', 'sep',
             'oct', 'nov', 'dec']


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


df = pd.read_csv(r'C:\Users\16028\OneDrive\Documents\prec_example.txt',
                 delim_whitespace=True, names= col_names, converters={'id': lambda x: str(x)})

    
df['state_code'] = df['id'].astype(str).str[:3]
df['div_number'] = df['id'].astype(str).str[3:4]
df['element_code'] = df['id'].astype(str).str[4:6]
df['year'] = df['id'].astype(str).str[6:10]


number_list = []
for i in range(1,51):
    number_list.append(str(i).zfill(3))
number_list
    

df['year'] = df['year'].astype(int)
df = df[df['year'] > 1970]

df = df[df['state_code'].isin(number_list)]
code_to_area_dict = getCodeToAreaDict()
area_to_code_dict = getAreaToCodeDict()



df['state'] = df['state_code'].map(code_to_area_dict)


col_names = ['id', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug', 'sep',
             'oct', 'nov', 'dec']
df_temp = df[col_names]


df_temp2 = df_temp.melt(id_vars='id')


sum_df = pd.read_csv(r'sum_df_v3.csv')


sum_df['state'] = sum_df['state'].str.lower()


month_to_number = monthToNumberDict()
df_temp2['month'] = df_temp2['variable'].map(month_to_number)

#######################
df = df_temp2.copy()

df['year'] = df['id'].astype(str).str[6:10]
df['year'] = df['year'].astype(int)
df['state_code'] = df['id'].astype(str).str[:3]
df['state'] = df['state_code'].map(code_to_area_dict)


df_new = pd.merge(df, sum_df, on =['year', 'month', 'state'], how='inner')
df_new

temp = df_new[:10000]


df_new.to_csv(r'C:\Users\16028\Downloads\storm_figuring_out_stuff.csv')


alist = [i for i i]

def getPreviousTen(row):
    curr_year = row['year']
    state = row['state']
    prev_ten = [i for i in range(row['year'] - 11, row['year'])]
    avg_value = df[(df['state'] == state) & (df['year'].isin(prev_ten))]['value'].mean()
    return avg_value
    
df['prev_ten_avg'] = df.apply(getPreviousTen)

df.columns
df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1)).dt.date





df['last_year_average'] = df[df['year'] == ]

df['new_month'] = df['value']

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score



def createOldFeatures():
data2 = df.groupby(['year', 'state', 'month'])
    
    week_date = split_date - timedelta(days=7)
    two_week_date = split_date - timedelta(days=14)
    month_date = split_date - timedelta(days=30)
    two_month_date = split_date - timedelta(days=60)
    three_month_date = split_date - timedelta(days=90)
    dfdict = pd.DataFrame(columns = ['device_id'])

    
    mylists = []
    for name, group in data2:
        if group.shape[0] < 50:
            continue
        date_seq = group[group['date'] < two_week_date].shape[0] #2020, 12, 1
        percent_before_last_two_weeks = date_seq / max(group[group['date'] < split_date].shape[0],1) #dt.date(2020,12,15)
        
        
        # age_seq2 = ','.join((list(group['Age'].astype(str))))
        total_alarms = group.shape[0]
        total_in_last_week = group[group.date > week_date].shape[0]
        total_in_last_month = group[group.date > month_date].shape[0]
        total_ratio_week_over_month = total_in_last_week/max(total_in_last_month,1)
        total_30_over_90 = total_in_last_month/ max(group[group.date > three_month_date].shape[0],1)
        ################################################################################################
        total_in_two_weeks = group[group.date > two_week_date].shape[0]                            
        total_in_two_months = group[group.date > two_month_date].shape[0]
        total_two_week_over_60 = total_in_last_month/max(group[group.date > two_month_date].shape[0],1)
        
        moderate_in_two_weeks = group[(group.pri == '21') & (group.date > two_week_date)].shape[0]                            
        moderate_in_two_months = group[(group.pri == '21') & (group.date > two_month_date)].shape[0]   
        moderate_two_week_over_60 = moderate_in_two_weeks/max(moderate_in_two_months,1)
        
        
        one_over_11_in_two_weeks = group[(group.priority == '1') & (group.date > two_week_date)].shape[0]/max(group[(group.priority == '11') & (group.date > week_date)].shape[0],1)
        one_over_11_in_two_months = group[(group.priority == '1') & (group.date > two_month_date)].shape[0]/max(group[(group.priority == '11') & (group.date > week_date)].shape[0],1)
        one_over_11_two_week_over_60 = one_over_11_in_two_weeks/max(one_over_11_in_two_months,1)
        
        
        eighty_seven_total = group[group.priority == '87'].shape[0]
        eighty_seven_in_last_week = group[(group.priority == '87') & (group.date > week_date)].shape[0]
        eighty_seven_total_in_last_month = group[(group.priority == '87') & (group.date > month_date)].shape[0]
        eighty_eight_total = group[group.priority == '88'].shape[0]
        eighty_eight_in_last_week = group[(group.priority == '88') & (group.date > week_date)].shape[0]
        eighty_eight_in_last_month = group[(group.priority == '88') & (group.date > month_date)].shape[0]
        eight_seven_over_eighty_eight_total = eighty_seven_total/max(eighty_eight_total,1)
        eight_seven_over_eighty_eight_in_last_week = eighty_seven_in_last_week/max(eighty_eight_in_last_week,1)
        
        DMXPLR = group[group.DMXPLR == 1].shape[0]/total_alarms
        DMXTEND = group[group.DMXTEND == 1].shape[0]/total_alarms
        DMXTND = group[group.DMXTND == 1].shape[0]/total_alarms
        # MPR = group[group.MPR == 1].shape[0]/total_alarms
        TSS5 = group[group.TSS5 == 1].shape[0]/total_alarms
        WDMX = group[group.WDMX == 1].shape[0]/total_alarms
        DMX = group[group.DMX == 1].shape[0]/total_alarms
        DMX_like = (total_alarms - WDMX)/total_alarms
        
        ################################################################################################
        
        moderate_total = group[group.pri == '21'].shape[0]
        moderate_in_last_week = group[(group.pri == '21') & (group.date > week_date)].shape[0]
        moderate_in_last_month = group[(group.pri == '21') & (group.date > month_date)].shape[0]
        moderate_ratio_week_over_month = moderate_in_last_week / max(moderate_in_last_month, 1)
        moderate_over_total = moderate_total/max(total_alarms,1)
        moderate_over_total_in_last_week = moderate_in_last_week/max(total_in_last_week,1)
        moderate_30_over_90 = moderate_in_last_month/ max(group[(group.pri == '1') & (group.date > three_month_date)].shape[0],1)
        
        
        critical_total = group[group.pri == '1'].shape[0]
        critical_in_last_week = group[(group.pri == '1') & (group.date > week_date)].shape[0]
        critical_in_last_month = group[(group.pri == '1') & (group.date > month_date)].shape[0]
        critical_ratio_week_over_month = critical_in_last_week / max(critical_in_last_month, 1)
        critical_over_total = critical_total/max(total_alarms,1)
        critical_over_moderate = critical_total/max(moderate_total,1)
        critical_over_total_in_last_week = critical_in_last_week/max(total_in_last_week,1)
        critical_30_over_90 = critical_in_last_month/ max(group[(group.pri == '1') & (group.date > three_month_date)].shape[0],1)
        
        one_over_11_total = group[group.priority == '1'].shape[0]/max(group[group.priority == '11'].shape[0],1)
        one_over_11_in_last_week = group[(group.priority == '1') & (group.date > week_date)].shape[0]/max(group[(group.priority == '11') & (group.date > week_date)].shape[0],1)
        one_over_11_in_last_month = group[(group.priority == '1') & (group.date > week_date)].shape[0]/max(group[(group.priority == '11') & (group.date > month_date)].shape[0],1)
        one_over_11_ratio_week_over_month = one_over_11_in_last_week/max(one_over_11_in_last_month,1)
        
        #shoul
        a1 = group[group.a1 == 1].shape[0]
        a2 = group[group.a2 == 1].shape[0]
        a3 = group[group.a3 == 1].shape[0]
        a4 = group[group.a4 == 1].shape[0]
        a5 = group[group.a5 == 1].shape[0]
        a6 = group[group.a6 == 1].shape[0]
        a7 = group[group.a7 == 1].shape[0]
        a_none = group[group['None'] == 1].shape[0]
        
# =============================================================================
#         a8 = group[group.a8 == 1].shape[0]
#         a9 = group[group.a9 == 1].shape[0]
#         a10 = group[group.a10 == 1].shape[0]
#         a11 = group[group.a11 == 1].shape[0]
#         a12 = group[group.a12 == 1].shape[0]
#         
# =============================================================================
        
        a1_in_past_week = group[(group.a1 == 1) & (group.date > week_date)].shape[0]
        a2_in_past_week = group[(group.a2 == 1) & (group.date > week_date)].shape[0]
        a3_in_past_week = group[(group.a3 == 1) & (group.date > week_date)].shape[0]
        
        return