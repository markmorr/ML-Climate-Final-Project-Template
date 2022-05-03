# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:11:42 2022

@author: 16028
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt


#recent ~10 years
df2019 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')
df3 = df2019.copy()


listy_2010 = [val + 2010 for val in range(12)]
for i in listy_2010:
    # df_dict[i] = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d' + str(i) + '_c20210803.csv')
    df_dict[i] = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d' + str(i) + '_c20220217.csv')
        

df2017 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2017_c20220124')


df2019 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')
df2017 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2017_c20220124.csv')
df2016 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2016_c20211217.csv')

import glob

PATH = 'C:\\Users\\16028\\Downloads\\storm_details\\'
glob.glob(PATH)




for f in glob.glob(PATH):
    print(f)

#difference is the 2017 vs 2014 (maybe when I downloaded?)
df2 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2020_c20220214.csv')
df2_loc = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_locations-ftp_v1.0_d2020_c20220217.csv')
df2_det = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2020_c20220217.csv')
#extras

df2_loc.columns = df2_loc.columns.str.lower()
df2_det.columns = df2_det.columns.str.lower()

df2_merged = pd.merge(df2_loc, df2_det, on='event_id')

df_loc_1 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_locations-ftp_v1.0_d2021_c20220217.csv')
df_loc_2 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\2019\StormEvents_locations-ftp_v1.0_d2019_c20220214.csv')


#just take one of the modern years
df = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')


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


df.event_type.value_counts()

# df = df.sample(n=1000, random_state=1)
df.columns
cols_to_use = ['begin_yearmonth', 'begin_day', 'episode_id', 'event_id', 'state', 'event_type', 
         'magnitude', 'category', 'tor_f_scale', 'tor_length', 'tor_width', 'begin_azimuth', 'begin_lat', 'begin_lon',
         'damage_property']

cols_to_drop =['begin_azimuth', 'episode_id', 'category',]
df = df[cols_to_use]
df.drop(columns=cols_to_drop, inplace=True)

# https://stackoverflow.com/questions/39684548/convert-the-string-2-90k-to-2900-or-5-2m-to-5200000-in-pandas-dataframe
df.damage_property = (df.damage_property.replace(r'[KM]+$', '', regex=True).astype(float) * \
df.damage_property.str.extract(r'[\d\.]+([KM]+)', expand=False)
.fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))


df['damage_property'] = df['damage_property']/25 #gonna have to adjust for their rough estimate practice
df['damage_property'] = np.log(df['damage_property'] + 1) #log +1  to not have log(0)
plt.hist(df['damage_property'])

counts, bins = np.histogram(df.damage_property)
plt.hist(bins[:-1], bins, weights=counts)
plt.grid()
plt.show()



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
print(df.isna().sum())
df.dropna(inplace=True)
X = df.drop(columns=['damage_property'])
y = df['damage_property']

df.columns

def ConvertToClassification(x):
    if (x >= 0) and (x < 1):
        return 'low'
    elif x < 5:
        return 'medium'
    else:
        return 'high'
    
df['pd'] = df['damage_property'].apply(ConvertToClassification) 

df.pd.value_counts()
df.damage_property.value_counts()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)


from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=12, random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
df['damage_property'] = y
df.drop(columns=['pd'], inplace=True)
from sklearn.manifold import TSNE
import seaborn as sns

df.columns

 #something's going wrong there

# ============================================================================= 
# tsne = TSNE(n_components=2, random_state=0, learning_rate=100, init='pca', square_distances=True)
# data = tsne.fit_transform(df.drop(columns='damage_property'))
# data = tsne.fit_transform(df)
# plt.scatter(data[:,0], data[:,1], c=y, label='Cost of damages'  )
# plt.title('t-SNE of storms')
# plt.legend(loc='upper left')
# plt.show()
# sns.scatterplot(data[:,0], data[:,1], hue=y, legend='full',)   
#  
# =============================================================================



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



# export GOOGLE_API_KEY=<Secret API Key>
# export GOOGLE_CLIENT=<Secret Client>
# export GOOGLE_CLIENT_SECRET=<Secret Client Secret>

# =============================================================================
# from urllib2 import urlopen
# import json
# def getplace(lat, lon):
#     url = "http://maps.googleapis.com/maps/api/geocode/json?"
#     url += "latlng=%s,%s&sensor=false" % (lat, lon)
#     v = urlopen(url).read()
#     j = json.loads(v)
#     components = j['results'][0]['address_components']
#     country = town = None
#     for c in components:
#         if "country" in c['types']:
#             country = c['long_name']
#         if "postal_town" in c['types']:
#             town = c['long_name']
#     return town, country
# 
# =============================================================================

print(getplace(51.1, 0.1))
print(getplace(51.2, 0.1))
print(getplace(51.3, 0.1))



# admin.google.com is used for Google Workspace accounts only. Regular Gmail accounts cannot be used to sign in to admin.google.com
# https://stackoverflow.com/questions/20938728/google-developer-console-disabled



df = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')

df.head()
df.columns
df.columns = df.columns.str.lower()
df = df[cols_to_use]

import random
from random import choices
words = ['arizona', 'new mexico', 'connecticut']
my_state_list = choices(words, k = df.shape[0])
my_state_list = pd.Series(my_state_list)
df['state'] = my_state_list


from fips_code_dictionary import get_state_to_fips_dict
mydict = get_state_to_fips_dict()
df['state_fips_code'] = df['state'].str.upper().map(mydict)
df.state_fips_code


from vega_datasets import data
states = alt.topo_feature(data.us_10m.url, feature='states')

# Encode basemap to depict all states
basemap= alt.Chart(states).mark_geoshape(
        stroke='white',
        strokeWidth=1,
        fill='lightgray'
    ).properties(
        width=500,
        height=400
    ).project('albersUsa')

basemap

df['damage_cost'] = df['damage_property']
df.columns


ab = df.copy()
df = df[:4000]
storm_data_state = df.copy()
# Encode filled map to depict damage cost by states
storm_data_by_state = storm_data_state.groupby(['state','state_fips_code'])['damage_cost'].sum().reset_index(name='damage_cost')
storm_data_by_state['state_fips_code']=storm_data_by_state['state_fips_code'].str.lstrip("0")
statemap = alt.Chart(states).mark_geoshape(  
).encode(color=alt.Color('damage_cost:Q', scale=alt.Scale(scheme='greenblue'),
                         legend=alt.Legend(title = "Total damage Cost")),
         tooltip=['state:N','damage_cost:Q']         
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(storm_data_by_state, 'state_fips_code', ['damage_cost','state']) 

)

# Encode point map to depict no. of storms by state
pointmap = alt.Chart(storm_data_state).mark_circle().encode(
    longitude='state_longitude:Q',
    latitude='state_latitude:Q',
    size=alt.Size('sum(no_of_storms):Q', title='Number of Storms'),
    color=alt.Color('sum(no_of_storms):Q', scale=alt.Scale(scheme='blueorange'),
                legend=alt.Legend(orient='right')),
    tooltip=['state:N','sum(no_of_storms):Q', 'sum(damage_cost):Q']
)

costmap = basemap + statemap

# Layer the filled and point maps
alt.layer(costmap, pointmap).resolve_legend(
    color="independent",
    size="independent"
).resolve_scale(color="independent")


alt.Chart(storm_data_state).mark_point().encode(
    x='no_of_storms:Q',
    y='damage_cost:Q',
    color='event_type:N',    
    tooltip = ['state:N','event_type:N','no_of_storms:Q','damage_cost:Q']
).configure_point(
    size=100
)

alt.Chart.show()





import altair as alt

# load a simple dataset as a pandas DataFrame
from vega_datasets import data
cars = data.cars()

# https://altair-viz.github.io/user_guide/display_frontends.html

alt.renderers.enable('mimetype')

alt.Chart(cars).mark_point().encode(
    x='Horsepower',
    y='Miles_per_Gallon',
    color='Origin',
)




df = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')

df.head()
df.columns
df.columns = df.columns.str.lower()
df = df[cols_to_use]


df.damage_property = (df.damage_property.replace(r'[KMB]+$', '', regex=True).astype(float) * \
df.damage_property.str.extract(r'[\d\.]+([KMB]+)', expand=False)
.fillna(1).replace(['K','M', 'B'], [10**3, 10**6,10**6 ]).astype(int))
df['damage_cost'] = df['damage_property']
df.damage_cost.value_counts()
df.damage_cost.hist()

df['ay'] = np.log(df['damage_cost']+1)

df.ay.value_counts()
df.ay.hist()
