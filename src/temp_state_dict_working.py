# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:44:46 2022

@author: 16028
"""


def getStateList():
    state_list = [                
    "Alabama",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
    "Alaska",  
    "Northeast Region", 
    "East North Central Region",  
    "Central Region",
    "Southeast Region",
    "West North Central Region",  
    "South Region",
    "Southwest Region",
    "Northwest Region",
    "West Region",
    "National (contiguous States)"]
                               
    return state_list              
    
       
def getNumberList():                
    number_list = []
    for i in range(1,51):
        number_list.append(str(i).zfill(3))
    
    for i in range(101,111):
        number_list.append(str(i).zfill(3))
    return number_list
    
def getCodeToAreaDict():
    number_list = getNumberList()
    state_list = getStateList()
    code_to_area_dict = dict(zip(number_list, state_list))
    code_to_area_dict =  {k: v.lower() for k, v in code_to_area_dict.items()}
    return code_to_area_dict
    
    
def getAreaToCodeDict():
    number_list = getNumberList()
    state_list = getStateList()
    code_to_area_dict = dict(zip(number_list, state_list))
    code_to_area_dict =  {k: v.lower() for k, v in code_to_area_dict.items()}
    abbrev_to_us_state = dict(map(reversed, code_to_area_dict.items()))   
    return abbrev_to_us_state


def monthToNumberDict():
    mylist = [i for i in range(1,13)]
    month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'july', 'aug', 'sep',
                 'oct', 'nov', 'dec']

    mydict = dict(zip(month_list, mylist))
    mydict
    return mydict

    