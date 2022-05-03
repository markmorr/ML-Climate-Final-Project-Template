# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:20:59 2022

@author: 16028
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:55:04 2022

@author: 16028
"""

import pandas as pd
import glob


base_url = 'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/'
new_address_dict = {
"prec": "climdiv-pcpncy-v1.0.0-20220406",
"tmpc": "climdiv-tmpccy-v1.0.0-20220406"
}
# "tmax": # "climdiv-tmaxcy-v1.0.0-20220406"

col_names = ['id', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug', 'sep',
             'oct', 'nov', 'dec']

new_df_dict = {}
myguy = 'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-tmpcst-v1.0.0-20220406'
new_df_dict['tmpc'] = pd.read_csv(myguy, delim_whitespace=True, names= col_names, converters={'id': lambda x: str(x)})
    
for name, address in new_address_dict.items():
    print(name)
    print(address)
    new_df_dict[name] = pd.read_csv(base_url + address, delim_whitespace=True, 
                                    names= col_names, converters={'id': lambda x: str(x)})
    

for name, dfy in new_df_dict.items():
    dfy.to_csv(r'C:\Users\16028\Downloads\storm_figuring_out_stuff\local_noaa_county\raw_' + name + '.csv',
               index=False)


def get_df_dict_county():
    df_dict = dict()
    directory_path = r"C:/Users/16028/Downloads/storm_figuring_out_stuff/local_noaa_county"
    res = [f for f in glob.glob( directory_path + r"/" +'*.csv')]# or "123" in f or "a1b" in f]
    feature_category_list = ['prec', 'tmin', 'tmpc', 'tmax']
    # ' \ ' and ' / ' are interchangeable in python paths
    for filename in res:
    
        # print(filename)
        file_id = filename.split('raw_')[1]
        file_id = file_id.split('.csv')[0]
        # print(file_id)
        if file_id in feature_category_list:
            x = pd.read_csv(filename, converters={'id': lambda x: str(x)}) # delim_whitespace=True,
            x.reset_index(drop=True)
            df_dict[file_id] = x
    return df_dict

############### DEVELOPING A WAY TO MAP TO THE COUNTY
address = "climdiv-pcpndv-v1.0.0-20220406"
dftemp = pd.read_csv( base_url + address, delim_whitespace=True, 
                                names= col_names, converters={'id': lambda x: str(x)})

df = dftemp.copy()
var_name = 'prec'
df[var_name] = df[col_names].mean(axis=1)
df = df[['id', 'prec']]
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