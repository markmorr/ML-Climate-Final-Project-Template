# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 17:07:16 2022

@author: 16028
"""

import pandas as pd
import numpy as np
import glob
import datetime as dt
import math
import matplotlib.pyplot as plt


df2 = pd.read_csv(r'C:\Users\16028\Downloads\dryad_central_sierra\Whole_WY_1971.csv')


all_files = glob.glob(r"C:\Users\16028\Downloads\dryad_central_sierra\Whole_WY_[0-9][0-9][0-9][0-9].csv")
dryad_dfs = (pd.read_csv(f) for f in all_files)
df = pd.concat(dryad_dfs, ignore_index=True)


# =============================================================================
# The files each cover from the beginning of october to the end of september the following year
# 2019 - 1971 + 1 = 49
# 49*365 + math.floor(49/4) = 17897 (matches the number of rows)
# data integrity seems good but need to continue checking
# note--some years are leap years
# Want to construct a website with nice visuals of the physical processes and the conditions
# underlying precipitation snow vs. rain 
# project SNOW
# =============================================================================

df.rename(columns={'Snow Water Equivalent (cm)': 'swe'}, inplace=True)
df.swe.fillna(0, inplace=True)
df.swe.isna().sum()
df['swe'] = df['swe'].astype(int)
df.swe

