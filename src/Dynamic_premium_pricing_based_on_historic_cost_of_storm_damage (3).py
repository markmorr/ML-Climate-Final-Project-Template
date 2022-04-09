#!/usr/bin/env python
# coding: utf-8

# # Google Cloud Public Datasets Program
# 
# ## Dynamic premium pricing based on historic cost of storm damage
# 
# Climate change will continue to evolve the impact of severe storms around the world, making it increasingly more difficult to determine effective pricing for the areas that are most heavily impacted by natural disasters. By analyzing the geographical distribution of the storms and their impact over a period of time,  insurers can better understand the risk involved and factor it in their dynamic pricing of premiums. Dynamic Premium Pricing is a complex problem in practice, involving several factors of which risk based on storms can be a useful and significant one. 
# 
# This demo analyzes [Severe Storm Event data produced by NOAA](https://console.cloud.google.com/marketplace/details/noaa-public/severe-storm-events) available through the Google Cloud Public Datasets Program and can help you better understand the risk that severe storms present to insurance. The analysis uses [BigQuery Geographic Information Systems (GIS)](https://cloud.google.com/bigquery/docs/gis-intro) to combine the storm data with [US zip code boundary data](https://console.cloud.google.com/marketplace/details/united-states-census-bureau/us-geographic-boundaries) also available through the Cloud Public Datasets Program. The US boundary data is one of more than 10 datasets available in BigQuery that define boundaries in the US, including states, zip codes, counties, and more.
# 
# To see the entire catalog of datasets available through the Google Cloud Public Datasets Program, check out [GCP Marketplace](https://console.cloud.google.com/marketplace/browse?filter=solution-type:dataset&q=public%20dataset). To learn more about the Google Cloud Public Datasets Program, see our [landing page](https://cloud.google.com/public-datasets).
# 
# 
# 

# ## Getting Started
# 
# 
# 

# We will use Python 3 and Big Query to read, analyze, and visualize the data. Let's install and import the required packages.

# In[ ]:



get_ipython().system('pip install geopandas==0.8.1')
get_ipython().system('pip install pyshp')
get_ipython().system('pip install shapely')
get_ipython().system('pip install geoplot')
get_ipython().system('pip install altair')
get_ipython().system('pip install vega')
get_ipython().system('pip install gpdvega')
# Ignore any warnings


# In[ ]:


# %%bash
# conda info --envs
# source activate py38_google
# python


# In[ ]:



import sys
# some simple python commands
sys.path.append('/usr/local/lib/python3.6/site-packages')
print(sys.path)

print("Python version")
print(sys.version)


# In[ ]:


from google.cloud import bigquery
from google.colab import auth

import pandas as pd
import json
import geopandas as gpd
from shapely import wkt
from shapely import geometry
import matplotlib.pyplot as plt
import altair as alt
import gpdvega
import vega
import time


# We will take a look at the storms data briefly by reading only one month of data. We are using the bigquery magic command to read the data into a pandas dataframe.

# In[ ]:


auth.authenticate_user()
print('Authenticated')
PROJECT = '<YOUR PROJECT>' #TODO Replace with your GCP PROJECT
PROJECT = 'siree-project-69256'


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'storm_data_Jan_2019 --project $PROJECT', '\nSELECT  *\nFROM `bigquery-public-data.noaa_historic_severe_storms.storms_2019` \nWHERE \nEXTRACT(MONTH FROM event_begin_time)=1')


# In[ ]:



storm_data_Jan_2019.head(10)


# In[ ]:


storm_data_Jan_2019.head(10)


# ## Explore the storms and geographical data

# To start with, let's read the storm data for the past 5 years and look at the distribution of number of storms and damage costs by state and event type.

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'storm_data_state --project $PROJECT', '# Get Storm data by State for the past 5 years from BigQuery Pulic Dataset\nWITH\n  storm AS (\n  SELECT\n    state,\n    event_type,\n    COUNT(event_id) no_of_storms,\n    SUM(damage_property) AS damage_property,\n    SUM(damage_crops) AS damage_crops,\n    SUM(damage_property + damage_crops) AS damage_cost\n  FROM\n    `bigquery-public-data.noaa_historic_severe_storms.storms_*`\n  WHERE\n    EXTRACT(YEAR\n    FROM\n      event_begin_time)>= EXTRACT(YEAR\n    FROM\n      CURRENT_DATE())-5\n  GROUP BY\n    state,\n    event_type  )\nSELECT\n  storm.*,  \n  state_fips_code,\n  int_point_lat AS state_latitude,\n  int_point_lon AS state_longitude\nFROM\n  storm,\n  `bigquery-public-data.geo_us_boundaries.states`\nWHERE\n  storm.state=state_name')


# In[ ]:


# Plot no. of storms and cost of damange on US map 

# Get States data from vega-datasets for mapping
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


# In[ ]:


# Layer the filled and point maps
alt.layer(costmap, pointmap).resolve_legend(
    color="independent",
    size="independent"
).resolve_scale(color="independent")


# As we can observe from the above, Texas has by far the highest number of storms and also  the most damage to its properties and crops. Texas is followed by California, Florida and Lousiana in terms of the amount of damage. Even though Kansas, Iowa and Virginia have more number of storms hitting them, the amount of damage is not really proportional. It could be either because there is less property/crops to be affected or because the severity of these storms is lower than the states with the lesser number of storms but with greater damage experienced.  
# 
# The below scatterplot clealry shows that higher no of storms doesn't always result in higher damages. We can explore further by looking at the type of storms and whether it affects the impact.

# In[ ]:


alt.Chart(storm_data_state).mark_point().encode(
    x='no_of_storms:Q',
    y='damage_cost:Q',
    color='event_type:N',    
    tooltip = ['state:N','event_type:N','no_of_storms:Q','damage_cost:Q']
).configure_point(
    size=100
)


# Hmm, there are way too many event types to derive any meaningful insight of their relative impact. Let's visualize storm types by the amount of damage they have resulted in. We may be able to identify a few top storm types that cause the most damage as per the Pareto rule. 

# In[ ]:


alt.Chart(storm_data_state).transform_aggregate(
    damage = 'sum(damage_cost)',
    groupby = ['event_type']
).transform_window(
    rank='rank(damage)',
    sort=[alt.SortField('damage', order='descending')]
).transform_filter(
    alt.datum.rank <=10
).mark_bar().encode(
    alt.X('damage:Q', title='Total Damage Cost'), 
    alt.Y('event_type:N', sort='-x'),       
    color=alt.value('green'),    
    tooltip = ['event_type:N','sum(no_of_storms):Q','sum(damage_cost):Q']
)


# 
# Clearly, there are only a handful of storm types that are most destructive. Based on the above, we can narrow down our dataset of interest to the states and event types that result in most damage. 

# In[ ]:


storm_data_state_subset = storm_data_state[storm_data_state['state'].isin(['California','Texas','Lousiana','Florida','Georgia'])]
storm_data_state_subset = storm_data_state_subset[storm_data_state_subset['event_type'].isin(['flash flood','wildfire','hurricane','flood','hail','tornado','tropical storm','storm surge/tide','high wind','thunderstorm wind'])]


# In[ ]:


alt.Chart(storm_data_state_subset).mark_point().encode(
    x='no_of_storms:Q',
    y='damage_cost:Q',
    color='event_type:N',    
    tooltip = ['state:N','event_type:N','no_of_storms:Q','damage_cost:Q']
).configure_point(
    size=100
)


# Now, let's dig deeper into the dataset by analyzing the top states of interest - Texas, California, Lousiana, Georgia and Florida to see how their impact varies by zip code. This would enable to calcuate the risk factor at a more granular level based on the insured's zip code.
# 
# We need to query the zip code data from Big Query by determining which zip code each event falls under. We can leverage Big Query's GIS functions to do this in SQL itself.

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'storm_data_zipcodes --project $PROJECT', "# Storm data by State for the past 5 years\n\n  SELECT\n    state,\n    zip_code,\n    internal_point_lat AS zip_code_latitude,\n    internal_point_lon AS zip_code_longitude,\n    COUNT(event_id) no_of_storms,\n    SUM(damage_property) AS damage_property,\n    SUM(damage_crops) AS damage_crops,\n    SUM(damage_property + damage_crops) AS damage_cost\n  FROM\n    `bigquery-public-data.noaa_historic_severe_storms.storms_*`,\n    `bigquery-public-data.geo_us_boundaries.zip_codes` AS zipcodes\n  WHERE\n    EXTRACT(YEAR\n    FROM\n      event_begin_time)>= EXTRACT(YEAR\n    FROM\n      CURRENT_DATE())-5\n    AND ST_WITHIN(event_point,zip_code_geom)\n    AND state IN ('California','Texas','Louisiana','Georgia','Florida')  \n    AND event_type IN ('flash flood','wildfire','hurricane','flood','hail','tornado','tropical storm','storm surge/tide','high wind','thunderstorm wind')\n  GROUP BY\n    state,\n    zip_code,\n    internal_point_lat,\n    internal_point_lon  \n  \n\n ")


# Let's look at how these zip codes are distributed by number of storms and their impact in terms of damages to property and/or crops.

# In[ ]:



alt.Chart(storm_data_zipcodes).mark_point().encode(
    x='no_of_storms:Q',
    y='damage_cost:Q',
    color=alt.Color('state:N',scale=alt.Scale(scheme='category10')),
    #size=alt.Size('Hours per Month to Afford a Home:Q'),
    tooltip = ['zip_code:N', 'state:N','no_of_storms:Q','damage_cost:Q']
).configure_point(
    size=100
)


# There are definitely some locations in Texas and Lousiana which have a higher risk profile than the rest. Viewing it on the map

# In[ ]:


zmap = alt.Chart(storm_data_zipcodes).mark_circle().encode(
    longitude='zip_code_longitude:Q',
    latitude='zip_code_latitude:Q',
    size=alt.Size('sum(damage_cost):Q', title=''),
    color=alt.Color('sum(damage_cost):Q', scale=alt.Scale(scheme='oranges'),
                legend=alt.Legend(orient='right', title='Damage Cost by Zip Code')),
    tooltip=['state:N','zip_code:N','sum(no_of_storms):Q', 'sum(damage_cost):Q']
)

alt.layer(basemap, zmap).resolve_legend(
    color="independent",
    size="independent"
).resolve_scale(color="independent")


# To illustrate the point that the patterns and risk changes over time, let's quickly look at the data for the top states from 20 years ago.

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'storm_data_zipcodes_20yearsago --project $PROJECT', "# Get Storm data by Zip Code for the past 20 years\n\n  SELECT\n    state,\n    zip_code,\n    internal_point_lat AS zip_code_latitude,\n    internal_point_lon AS zip_code_longitude,\n    COUNT(event_id) no_of_storms,\n    SUM(damage_property) AS damage_property,\n    SUM(damage_crops) AS damage_crops,\n    SUM(damage_property + damage_crops) AS damage_cost\n  FROM\n    `bigquery-public-data.noaa_historic_severe_storms.storms_*`,\n    `bigquery-public-data.geo_us_boundaries.zip_codes` AS zipcodes\n  WHERE\n    EXTRACT(YEAR\n    FROM\n      event_begin_time)>= EXTRACT(YEAR\n    FROM\n      CURRENT_DATE())-25\n    AND EXTRACT(YEAR\n    FROM\n      event_begin_time)<= EXTRACT(YEAR\n    FROM\n      CURRENT_DATE())-20\n    AND ST_WITHIN(event_point,zip_code_geom)\n    AND state IN ('California','Texas','Louisiana','Georgia','Florida')  \n    AND event_type IN ('flash flood','wildfire','hurricane','flood','hail','tornado','tropical storm','storm surge/tide','high wind','thunderstorm wind')\n  GROUP BY\n    state,\n    zip_code,\n    internal_point_lat,\n    internal_point_lon  ")


# Plot the damage cost and number of storms for the top states 20 years ago

# In[ ]:


zmap_20 = alt.Chart(storm_data_zipcodes_20yearsago).mark_circle().encode(
    longitude='zip_code_longitude:Q',
    latitude='zip_code_latitude:Q',
    size=alt.Size('sum(damage_cost):Q', title=''),
    color=alt.Color('sum(damage_cost):Q', scale=alt.Scale(scheme='teals'),
                legend=alt.Legend(orient='right', title='Damage Cost by Zip Code')),
    tooltip=['state:N','zip_code:N','sum(no_of_storms):Q', 'sum(damage_cost):Q']
)

alt.layer(basemap, zmap_20).resolve_legend(
    color="independent",
    size="independent"
).resolve_scale(color="independent")


# As you can see the the distribution of damage cost was very different 20 years ago for the selected states. 

# The exploration of data and the patterns we identified so far gave us a fair sense of weighing in the risk factor for various locations based on the historical storm data. However, as we have seen, the data changes over time and so are the patterns of risk profiles of various locations, which calls for repeated and complex analysis over the time. Also,deriving the insights at the most granular level at scale proves challenging with ad hoc analysis.
# 
# Let's try to build a machine leaning model that can group zip codes into logical risk profile clusters based on the storm data. This way, we can retrain the model as and when we get new data and reap the benefits of automatic categorization with fair accuracy.

# # BigQuery ML to cluster the zip codes based on cost of damages to property and crops
# 
# 

# ### Create Dataset
# 
# You'll need a BigQuery dataset to create the ML model in.

# In[ ]:


# Create dataset
client = bigquery.Client(project = PROJECT)
dataset_name = "insurance_demo"
dataset_id = "{}.{}".format(client.project, dataset_name)

# Construct a full Dataset object to send to the API.
dataset = bigquery.Dataset(dataset_id)

# Send the dataset to the API for creation.
# Raises google.api_core.exceptions.Conflict if the Dataset already
# exists within the project.
dataset = client.create_dataset(dataset)  # API request
print("Created dataset {}.{}".format(client.project, dataset.dataset_id))


# ### Create ML model
# Let's create a K-Mean clustering model with 6 clusters of zipcodes based on number of storms, event type, damage cost. 

# In[ ]:


get_ipython().run_cell_magic('bigquery', '--project $PROJECT', "\nCREATE OR REPLACE MODEL\n    `insurance_demo.bqml_storm_clusters` OPTIONS(model_type='kmeans', num_clusters=6)\nAS\n\nWITH zipcodes\nAS\n(\nSELECT\n    state,\n    zip_code,\n    event_type,\n    COUNT(event_id) no_of_storms,\n    SUM(damage_property) AS damage_property,\n    SUM(damage_crops) AS damage_crops,\n    SUM(damage_property + damage_crops) AS damage_cost\n  FROM\n    `bigquery-public-data.noaa_historic_severe_storms.storms_*`,\n    `bigquery-public-data.geo_us_boundaries.zip_codes` AS zipcodes\n  WHERE\n    EXTRACT(YEAR\n    FROM\n      event_begin_time)>= EXTRACT(YEAR\n    FROM\n      CURRENT_DATE())-5\n    AND ST_WITHIN(event_point,zip_code_geom)\n    AND state IN ('California','Texas','Louisiana','Georgia','Florida')  \n  GROUP BY\n    state,\n    zip_code,\n    event_type\n)\nSELECT event_type, no_of_storms, damage_property, damage_crops\nFROM zipcodes")


# Let's now evaluate the model to see how it performs

# In[ ]:


get_ipython().run_cell_magic('bigquery', '--project $PROJECT', '\nSELECT\n  *\nFROM\n  ML.EVALUATE(MODEL `insurance_demo.bqml_storm_clusters`)')


# The DB index is fairly low for the model to be acceptable. Please feel free to experiment with the number of clusters and input features to fine tune the model as needed.
# 
# Now, let's evaluate the model by providing Georgia data as input.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '--project $PROJECT', "\nSELECT\n  *\nFROM\n  ML.EVALUATE(MODEL `insurance_demo.bqml_storm_clusters`\n,\n    (\n    SELECT\n    state,\n    zip_code,\n    event_type,\n    COUNT(event_id) no_of_storms,\n    SUM(damage_property) AS damage_property,\n    SUM(damage_crops) AS damage_crops,\n    SUM(damage_property + damage_crops) AS damage_cost\n  FROM\n    `bigquery-public-data.noaa_historic_severe_storms.storms_*`,\n    `bigquery-public-data.geo_us_boundaries.zip_codes` AS zipcodes\n  WHERE\n    EXTRACT(YEAR\n    FROM\n      event_begin_time)>= EXTRACT(YEAR\n    FROM\n      CURRENT_DATE())-5\n    AND ST_WITHIN(event_point,zip_code_geom)\n    AND state IN ('Georgia')  \n  GROUP BY\n    state,\n    zip_code,\n    event_type)\n             )")


# Let's now predict the clusters based on the model.

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'predicted --project $PROJECT', "\nSELECT\n  CENTROID_ID,\n  state,\n  zip_code,\n  sum(no_of_storms) no_of_storms,\n  sum(damage_property) damage_property,\n  sum(damage_crops) damage_crops,\n  sum(damage_cost) damage_cost\nFROM\n  ML.PREDICT( MODEL insurance_demo.bqml_storm_clusters,\n    (\n   SELECT\n    state,\n    zip_code,\n    event_type,\n    COUNT(event_id) no_of_storms,\n    SUM(damage_property) AS damage_property,\n    SUM(damage_crops) AS damage_crops,\n    SUM(damage_property + damage_crops) AS damage_cost\n  FROM\n    `bigquery-public-data.noaa_historic_severe_storms.storms_*`,\n    `bigquery-public-data.geo_us_boundaries.zip_codes` AS zipcodes\n  WHERE\n    EXTRACT(YEAR\n    FROM\n      event_begin_time)>= EXTRACT(YEAR\n    FROM\n      CURRENT_DATE())-5\n    AND ST_WITHIN(event_point,zip_code_geom)\n    AND state IN ('California','Texas','Louisiana','Georgia','Florida')  \n  GROUP BY\n    state,\n    zip_code,\n    event_type\n    )\n    \n            )\n    GROUP BY  \n        CENTROID_ID,\n        state,\n        zip_code")


# In[ ]:


predicted.head()


# We can now look at the clusters and see if they are meaningful.

# In[ ]:


# To bypass the max rows limitation of 5000 for Altair
from altair import pipe, limit_rows, to_values
t = lambda data: pipe(data, limit_rows(max_rows=10000), to_values)
alt.data_transformers.register('custom', t)
alt.data_transformers.enable('custom')


# In[ ]:



alt.Chart(predicted).mark_circle().encode(
    x='no_of_storms:Q',
    y='damage_cost:Q',
    color=alt.Color('CENTROID_ID:N',scale=alt.Scale(scheme='dark2'), legend=alt.Legend(title='Cluster')),
    tooltip = ['CENTROID_ID:N','zip_code:N', 'state:N', 'sum(no_of_storms):Q','sum(damage_cost):Q']
).configure_circle(
    size=200,
    opacity=0.4
).interactive().properties(
    width=500,
    height=300
)


# The clusters of zip codes do indicate some resonable patterns, that can be translated to different risk profiles. Looking at one state - California, for example, help identify variance within a state.

# In[ ]:


#Look at one state at a time
alt.Chart(predicted[predicted['state']=='California']).mark_circle().encode(
    x='no_of_storms:Q',
    y='damage_cost:Q',
    color=alt.Color('CENTROID_ID:N',scale=alt.Scale(scheme='dark2'), legend=alt.Legend(title='Cluster')),   
    tooltip = ['CENTROID_ID:N','zip_code:N','no_of_storms:Q','damage_cost:Q']
).configure_circle(
    size=200,
    opacity=0.4
).interactive().properties(
    width=500,
    height=300
)


# We can extend the above to interpret the distribution of clusters or risk profiles for different states.

# In[ ]:


#Risk distribution for the top states
alt.Chart(predicted).transform_aggregate(
    no_of_storms='sum(no_of_storms)',
    damage_cost='sum(damage_cost)',
    groupby=['state','CENTROID_ID']
).mark_bar().encode(
    y='state:N',
    x='sum(damage_cost):Q',
    color=alt.Color('CENTROID_ID:N',scale=alt.Scale(scheme='category10')),    
    tooltip = ['CENTROID_ID:N', 'state:N','sum(no_of_storms):Q','sum(damage_cost):Q']
).properties(
    width=500,
    height=250
)


# ## Summary
# 
# By analyzing the historical storm data and combining it with US geography data, we could deduce the risk associated with these natural events that can be used for calcuating more effective dynamic pricing for insurance premiums.
# 
# You used storm data from the [Google Cloud Public Datasets Program](https://cloud.google.com/public-datasets/) to analyze analyze the risk factor assocaited with various US zip codes for dynamic premium pricing of insurance. You didn't have to discover, access, or onboard the dataset. You simply joined data already available in BigQuery through the Cloud Public Datasets Program and analyzed the data. This allowed you to get new, data-driven answers to your business questions significantly faster.
# 
# You can check out the [Public Datasets Program's data catalog](g.co/cloud/marketplace-datasets) to find other datasets that can help you learn more about your existing data and uncover new insights. 
# 
# 

# ## Cleanup
# 
# Delete the machine learning model

# In[ ]:


# Construct a BigQuery client object.
client = bigquery.Client(project = PROJECT)

model_id = PROJECT +'.insurance_demo.bqml_storm_clusters'
client.delete_model(model_id)  

print("Deleted model '{}'.".format(model_id))


# Delete the dataset 

# In[ ]:


client.delete_dataset(
    'insurance_demo', delete_contents=True, not_found_ok=True
) 
print("Deleted dataset insurance_demo")

