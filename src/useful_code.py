# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 16:55:05 2022

@author: 16028
"""

def genDatetimeFeatures(df):
    df['datetime'] = pd.to_datetime(df['feature_0'], format='%m-%d %X')
    min_day = dt(1900, 1, 1, 00, 00)
    df['days_since'] = (df['datetime'] - min_day).dt.days
    df["m"] = df.datetime.dt.month
    df["d"] = df.datetime.dt.day
    df["h"] = df.datetime.dt.hour
    df['day_of_week'] = df['days_since'] & 7
    return df


def runOneHot(df, feature_name):
    one_hot = pd.get_dummies(df[feature_name], prefix=feature_name)
    df = df.drop(feature_name,axis = 1)
    df = df.join(one_hot)
    return df

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
    
    
def mapToSeason(x):
    if x in [1,2,3]: #['jan', 'feb', 'mar']:
        return 'winter'
    elif x in [4,5,6]: #['apr', 'may', 'jun']:
        return 'spring'
    elif x in [7,8,9]: #['july', 'aug', 'sep']:
        return 'summer'
    elif x in [10,11,12]: #['oct', 'nov', 'dec']:
        return 'fall'


# state_to_region =  {k.lower(): v for k, v in state_to_region.items()}
# df['region'] = df['state'].map(state_to_region)
# abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))




# =============================================================================
# one_hot_cols = ['state'] #'event_type', 'tor_f_scale']
# one_hot = pd.get_dummies(df[one_hot_cols]) 
# df = df.drop(one_hot_cols, axis = 1)
# df = pd.concat([df, one_hot], axis=1)
# =============================================================================











