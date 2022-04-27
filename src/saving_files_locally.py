# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:55:04 2022

@author: 16028
"""

import pandas as pd


col_names = ['ghcn_id', 'lat', 'lon', 'stdv', 'yr_mo', 'network']
base_url = 'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/'

new_address_dict = { 
"cdd": "climdiv-cddcst-v1.0.0-20220406",
"hdd": "climdiv-hddcst-v1.0.0-20220406",
"prec": "climdiv-pcpnst-v1.0.0-20220406",
"pdsi": "climdiv-pdsist-v1.0.0-20220406",
"phdi": "climdiv-phdist-v1.0.0-20220406",
"pmdi": "climdiv-pmdist-v1.0.0-20220406",
"sp01": "climdiv-sp01st-v1.0.0-20220406",
"sp02": "climdiv-sp02st-v1.0.0-20220406",
"sp03": "climdiv-sp03st-v1.0.0-20220406",
"sp04": "climdiv-sp06st-v1.0.0-20220406",
"sp09": "climdiv-sp09st-v1.0.0-20220406",
"sp12": "climdiv-sp12st-v1.0.0-20220406",
"sp24": "climdiv-sp24st-v1.0.0-20220406",
"tmax": "climdiv-tmaxst-v1.0.0-20220406",
"tmin": "climdiv-tminst-v1.0.0-20220406",
"tmpc": "climdiv-tmpcst-v1.0.0-20220406",
"zndx": "climdiv-zndxst-v1.0.0-20220406",
}

print(new_address_dict['zndx'])

new_df_dict = {}
myguy = 'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-tmpcst-v1.0.0-20220406'
new_df_dict['tmpc'] = pd.read_csv(myguy, delim_whitespace=True, names= col_names, converters={'id': lambda x: str(x)})
    
for name, address in new_address_dict.items():
    print(name)
    print(address)
    new_df_dict[name] = pd.read_csv(base_url + address, delim_whitespace=True, names= col_names, converters={'id': lambda x: str(x)})
    

for name, dfy in new_df_dict.items():
    dfy.to_csv(r'C:\Users\16028\Downloads\storm_figuring_out_stuff\raw_' + name + '.csv')
    
# =============================================================================
# for name, addresss in new_address_dict.items():
#     print(base_url)
#     print(address)
#     print(base_url + address)
#     new_df_dict[name] = pd.read_csv(address, delim_whitespace=True, names= col_names, converters={'id': lambda x: str(x)})
#     
# =============================================================================
    