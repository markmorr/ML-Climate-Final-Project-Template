# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 19:28:53 2022

@author: 16028
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def correctDamageProp(x):
    if x == "K":
        return '1000'
    elif x == "M":
        return '1000000'
    elif x == "B":
        return '1000000'
    elif x == ".K":
        return '1000'
    elif x == ".M":
        return '100000'
    elif x == ".B":
        return '1000000'
    elif x == "K.":
        return '1000'
    elif x == "M.":
        return '100000'
    elif x == "B.":
        return '1000000'
    else:
        return x

#TODO:
    #Sophisticate the 

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
tempo = glued_data.sample(100000) 

first_tenth = glued_data[:100000]   






df1 = glued_data[:1000000]
df


# myseries = glued_data['DAMAGE_PROPERTY']



df = first_tenth.copy()
df = glued_data.copy()
# df = tempo.copy()

percent_missing = df.isnull().sum() * 100 / len(df)
print(percent_missing)  

df.columns = df.columns.str.lower()
cols_to_use = ['begin_yearmonth', 'begin_day', 'episode_id', 'event_id', 'state', 'event_type', 
         'magnitude', 'category', 'tor_f_scale', 'tor_length', 'tor_width', 'begin_azimuth',
         'begin_lat', 'begin_lon', 'damage_property']

cols_to_drop =['begin_azimuth', 'episode_id', 'category',]
df = df[cols_to_use]
df.drop(columns=cols_to_drop, inplace=True)



# df['damage_property'] = df['damage_property'].str.replace(' ', '')
# df['damage_property'] = df['damage_property'].str.replace('', '0')
df['damage_property'] = df['damage_property'].replace(r'^\s+$', np.nan, regex=True)
df['damage_property'].fillna('0', inplace=True)  
 
yg = df['damage_property'].value_counts()
print(yg)


df['damage_property'] = df['damage_property'].apply(correctDamageProp)


df.damage_property = (df.damage_property.replace(r'[KMB]+$', '', regex=True).astype(float) * \
df.damage_property.str.extract(r'[\d\.]+([KMB]+)', expand=False)
.fillna(1).replace(['K','M', 'B'], [10**3, 10**6,10**9 ]).astype(int))
df['damage_cost'] = df['damage_property']


interimDataPath = r'C:\Users\16028\Downloads\storm_details'
df['damage_cost'].to_csv(interimDataPath + '\saving_damage_corrected.csv')


# df.damage_cost.value_counts()
# df.damage_cost.hist()

df.damage_property.isna().sum()




 
# myseries = myseries.apply(correctDamageProp)
# myseries.isna().sum()
# =============================================================================
# df.reset_index(drop=True, inplace=True)
# from inspect import currentframe, getframeinfo
# pd.options.mode.chained_assignment = 'raise'
# 
# try:
#     df.damage_property = (df.damage_property.replace(r'[KMB]+$', '', regex=True).astype(float) * \
#     df.damage_property.str.extract(r'[\d\.]+([KMB]+)', expand=False)
#     .fillna(1).replace(['K','M', 'B'], [10**3, 10**6,10**9 ]).astype(int))
# except ValueError:
#     print('handling..')
#     frameinfo = getframeinfo(currentframe())
#     print(frameinfo.lineno)
# 
# 
# =============================================================================






print(5)

df.damage_property = (df.damage_property.replace(r'[KMB]+$', '', regex=True).astype(float) * \
df.damage_property.str.extract(r'[\d\.]+([KMB]+)', expand=False)
.fillna(1).replace(['K','M', 'B'], [10**3, 10**6,10**9 ]).astype(int))
df['damage_cost'] = df['damage_property']
df.damage_cost.value_counts()
df.damage_cost.hist()

df['ay'] = np.log(df['damage_cost']+1)

df.ay.value_counts()
df.ay.hist()















df.damage_property = (df.damage_property.replace(r'[KMB]+$', '', regex=True).astype(float) * \
df.damage_property.str.extract(r'[\d\.]+([KMB]+)', expand=False)
.fillna(1).replace(['K','M', 'B'], [10**3, 10**6,10**6 ]).astype(int))
df['damage_cost'] = df['damage_property']
df.damage_cost.value_counts()
df.damage_cost.hist()


for i, item in enumerate(df['damage_property']):
   try:
      
    df.damage_property = (df.damage_property.replace(r'[KMB]+$', '', regex=True).astype(float) * \
    df.damage_property.str.extract(r'[\d\.]+([KMB]+)', expand=False)
    .fillna(1).replace(['K','M', 'B'], [10**3, 10**6,10**6 ]).astype(int))
   except ValueError:
      print('ERROR at index {}: {!r}'.format(i, item))


df['damage_property'] = df['damage_property']/25 #gonna have to adjust for their rough estimate practice
df['damage_property'] = np.log(df['damage_property'] + 1) #log +1  to not have log(0)
plt.hist(df['damage_property'])

counts, bins = np.histogram(df.damage_property)
plt.hist(bins[:-1], bins, weights=counts)
plt.grid()
plt.show()












#recent ~10 years
# =============================================================================
# df1 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2021_c20220214.csv')
# df2 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2020_c20220214.csv')
# df3 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')
# df4 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2018_c20220214.csv')
# df5 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2017_c20220214.csv')
# df6 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2016_c20220214.csv')
# df7 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2015_c20220214.csv')
# df8 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2014_c20220214.csv')
# 
# =============================================================================
#extras
# df_loc_1 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_locations-ftp_v1.0_d2021_c20220217.csv')
# df_loc_2 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\2019\StormEvents_locations-ftp_v1.0_d2019_c20220214.csv')


#just take one of the modern years
df = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')

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
# =============================================================================
# listy_2010 = [val + 2010 for val in range(12)]
# for i in listy_2010:
#     df_dict[i] = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d' + str(i) + '_c20210803.csv')
#     
# =============================================================================

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


#############################################################################################################
# df = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')

# df.head()
# df.columns
# df.columns = df.columns.str.lower()
# df = df[cols_to_use]

# # https://stackoverflow.com/questions/39684548/convert-the-string-2-90k-to-2900-or-5-2m-to-5200000-in-pandas-dataframe
# df.damage_property = (df.damage_property.replace(r'[KMB]+$', '', regex=True).astype(float) * \
# df.damage_property.str.extract(r'[\d\.]+([KMB]+)', expand=False)
# .fillna(1).replace(['K','M', 'B'], [10**3, 10**6,10**6 ]).astype(int))
# df['damage_cost'] = df['damage_property']
# df.damage_cost.value_counts()
# df.damage_cost.hist()
# #############################################################################################################

# df['damage_property'] = df['damage_property']/25 #gonna have to adjust for their rough estimate practice
# df['damage_property'] = np.log(df['damage_property'] + 1) #log +1  to not have log(0)
# plt.hist(df['damage_property'])

# counts, bins = np.histogram(df.damage_property)
# plt.hist(bins[:-1], bins, weights=counts)
# plt.grid()
# plt.show()



######################

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

####################


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

# print(getplace(51.1, 0.1))
# print(getplace(51.2, 0.1))
# print(getplace(51.3, 0.1))





#recent ~10 years
# =============================================================================
# df1 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2021_c20220214.csv')
# df2 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2020_c20220214.csv')
# df3 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')
# df4 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2018_c20220214.csv')
# df5 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2017_c20220214.csv')
# df6 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2016_c20220214.csv')
# df7 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2015_c20220214.csv')
# df8 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2014_c20220214.csv')
# 
# =============================================================================
#extras
# df_loc_1 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_locations-ftp_v1.0_d2021_c20220217.csv')
# df_loc_2 = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\2019\StormEvents_locations-ftp_v1.0_d2019_c20220214.csv')


#just take one of the modern years
df = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')

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
# =============================================================================
# listy_2010 = [val + 2010 for val in range(12)]
# for i in listy_2010:
#     df_dict[i] = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d' + str(i) + '_c20210803.csv')
#     
# =============================================================================

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


#############################################################################################################
# df = pd.read_csv(r'C:\Users\16028\Downloads\storm_details\StormEvents_details-ftp_v1.0_d2019_c20220214.csv')

# df.head()
# df.columns
# df.columns = df.columns.str.lower()
# df = df[cols_to_use]

# # https://stackoverflow.com/questions/39684548/convert-the-string-2-90k-to-2900-or-5-2m-to-5200000-in-pandas-dataframe
# df.damage_property = (df.damage_property.replace(r'[KMB]+$', '', regex=True).astype(float) * \
# df.damage_property.str.extract(r'[\d\.]+([KMB]+)', expand=False)
# .fillna(1).replace(['K','M', 'B'], [10**3, 10**6,10**6 ]).astype(int))
# df['damage_cost'] = df['damage_property']
# df.damage_cost.value_counts()
# df.damage_cost.hist()
# #############################################################################################################

# df['damage_property'] = df['damage_property']/25 #gonna have to adjust for their rough estimate practice
# df['damage_property'] = np.log(df['damage_property'] + 1) #log +1  to not have log(0)
# plt.hist(df['damage_property'])

# counts, bins = np.histogram(df.damage_property)
# plt.hist(bins[:-1], bins, weights=counts)
# plt.grid()
# plt.show()



######################





####################


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

# print(getplace(51.1, 0.1))
# print(getplace(51.2, 0.1))
# print(getplace(51.3, 0.1))

















    