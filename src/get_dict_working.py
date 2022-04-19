# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:44:46 2022

@author: 16028
"""


 Alabama
New York
 Arizona
North Carolina
 Arkansas
North Dakota
 California
Ohio
 Colorado
Oklahoma
 Connecticut
Oregon
 Delaware
Pennsylvania
 Florida
Rhode Island
 Georgia
South Carolina
 Idaho
South Dakota
 Illinois
Tennessee
 Indiana
Texas
 Iowa
Utah
 Kansas
Vermont
 Kentucky
Virginia
 Louisiana
Washington
 Maine
West Virginia
 Maryland
Wisconsin
 Massachusetts
Wyoming
 Michigan
Alaska

 Minnesota
Northeast Region 
 Mississippi
East North Central Region  
 Missouri
Central Region  
 Montana
Southeast Region  
 Nebraska
West North Central Region  
 Nevada
South Region  
 New Hampshire
Southwest Region  
 New Jersey
Northwest Region  
 New Mexico
West Region

National (contiguous  States)
                 
                 
                 


 Alabama          New York
 Arizona          North Carolina
 Arkansas         North Dakota
 California       Ohio
 Colorado         Oklahoma
 Connecticut      Oregon
 Delaware         Pennsylvania
 Florida          Rhode Island
 Georgia          South Carolina
 Idaho            South Dakota
 Illinois         Tennessee
 Indiana          Texas
 Iowa             Utah
 Kansas           Vermont
 Kentucky         Virginia
 Louisiana        Washington
 Maine            West Virginia
 Maryland         Wisconsin
 Massachusetts    Wyoming
 Michigan         Alaska     
 Minnesota        Northeast Region 
 Mississippi      East North Central Region  
 Missouri         Central Region  
 Montana          Southeast Region  
 Nebraska         West North Central Region  
 Nevada           South Region  
 New Hampshire    Southwest Region  
 New Jersey       Northwest Region  
 New Mexico       West Region
                 National (contiguous  States)


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
                           
          

       
                
number_list = []
for i in range(1,51):
    number_list.append(str(i).zfill(3))

for i in range(101,111):
    number_list.append(str(i).zfill(3))
    
    
code_to_area_dict = dict(zip(number_list, state_list))
code_to_area_dict =  {k: v.lower() for k, ;v in code_to_area_dict.items()}
# df['region'] = df['state'].map(state_to_region)


abbrev_to_us_state = dict(map(reversed, code_to_area_dict.items()))    


    