# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:39:10 2022

@author: 16028
"""

(12*21 + 11) * 47 #expected number of records if every state and month had a reading

# an example of the structure of merges where you only want to pick out specific columns
# df = pd.merge(df,df2[['Key_Column','Target_Column']],on='Key_Column', how='left') #an example of the structure


### CODE TO CONFIRM THAT TAKING THE AVERAGE OF NEIGHBORING STATES FEATURES IS WORKING AS I EXPECT IT TO
#validation--it's working
sum_total = 0
for state in state_to_neighbor_dict['alabama']:
    precy = df_to_use[(df_to_use['state'] == state) & (df_to_use['date'] == dt.date(2002,4,1))]['past_6_prec'].item()
    print(precy)
    sum_total += precy 
print(sum_total/len(state_to_neighbor_dict['alabama']))   
