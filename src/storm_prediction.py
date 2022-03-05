# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:11:42 2022

@author: 16028
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import altair as alt


#recent ~10 years
df1 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2021_c20220214.csv')
df2 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2020_c20220214.csv')
df3 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')
df4 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2018_c20220214.csv')
df5 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2017_c20220214.csv')
df6 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2016_c20220214.csv')
df7 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2015_c20220214.csv')
df8 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2014_c20220214.csv')

#extras
df_loc_1 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_locations-ftp_v1.0_d2021_c20220217.csv')
df_loc_2 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\2019\StormEvents_locations-ftp_v1.0_d2019_c20220214.csv')


#just take one of the modern years
df = df3.copy()

#old sections
df1950 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d1950_c20210803.csv')
df1951 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d1951_c20210803.csv')
df1952 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d1952_c20210803.csv')
df1953 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d1953_c20210803.csv')
df1954 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d1954_c20210803.csv')
df1955 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d1955_c20210803.csv')
df1956 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d1956_c20210803.csv')

df_dict = dict()
listy_1950 = [val + 1950 for val in range(11)]
for i in listy_1950:
    df_dict[i] = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d' + str(i) + '_c20210803.csv')
    
#issues with the read in notation
listy_2010 = [val + 2010 for val in range(12)]
for i in listy_2010:
    df_dict[i] = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d' + str(i) + '_c20210803.csv')
    

df = pd.concat([df_dict[i] for i in df_dict.keys()])
df.columns = df.columns.str.lower()
print(df.columns)



# df = df.sample(n=1000, random_state=1)
df.columns
df = df[['begin_yearmonth', 'begin_day', 'episode_id', 'event_id', 'state', 'event_type', 
         'magnitude', 'category', 'tor_f_scale', 'tor_length', 'tor_width', 'begin_azimuth', 'begin_lat', 'begin_lon',
         'damage_property']]
df.drop(columns=['begin_azimuth', 'episode_id', 'category',], inplace=True)

# https://stackoverflow.com/questions/39684548/convert-the-string-2-90k-to-2900-or-5-2m-to-5200000-in-pandas-dataframe
df.damage_property = (df.damage_property.replace(r'[KM]+$', '', regex=True).astype(float) * \
df.damage_property.str.extract(r'[\d\.]+([KM]+)', expand=False)
.fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))
    

df['damage_property'] = df['damage_property']/25 #gonna have to adjust for their rough estimate practice
df['damage_property'] = np.log(df['damage_property'] + 1) #log +1  to not have log(0)
plt.hist(df['damage_property'])

counts, bins = np.histogram(df.damage_property)
plt.hist(bins[:-1], bins, weights=counts)



df['year'] = df['begin_yearmonth'].astype(str).str[:4].astype(int)
df['month'] = df['begin_yearmonth'].astype(str).str[4:].astype(int)
df['years_since'] = df['year'] - 1950

df['days_since'] = df['years_since']*365 + df['month']*12 + df['begin_day']
df.days_since
one_hot = pd.get_dummies(df[['event_type', 'tor_f_scale']])

df = df.drop(['event_type', 'tor_f_scale'], axis = 1)
# Join the encoded df
df = pd.concat([df, one_hot], axis=1)

df.drop(columns=[ 'begin_yearmonth', 'begin_day', 'event_id', 'state', 'year', 'month',
                 'years_since',], inplace=True)
X = df.drop(columns=['damage_property'])
y = df['damage_property']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=12, random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

np.log(2.7)
# Layer the filled and point maps
#CITE FROM GOOGLE
# =============================================================================
# alt.layer(costmap, pointmap).resolve_legend(
#     color="independent",
#     size="independent"
# ).resolve_scale(color="independent")
# 
# 
# alt.Chart(storm_data_state).mark_point().encode(
#     x='no_of_storms:Q',
#     y='damage_cost:Q',
#     color='event_type:N',    
#     tooltip = ['state:N','event_type:N','no_of_storms:Q','damage_cost:Q']
# ).configure_point(
#     size=100
# )
# 
# =============================================================================










