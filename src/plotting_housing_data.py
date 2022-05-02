# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 18:50:26 2022

@author: 16028
"""


import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

# https://www.redfin.com/news/data-center/
# from plotly.offline import plot
# ...
# # plot(fig) #this works too
# installing this: conda install -c plotly plotly-orca
# and then having import plotly.io as pio
# import plotly.express as px
# pio.renderers.default='svg' would supposedly let me do it in terminal too

housing = pd.read_csv(r'data\state_market_tracker.tsv000', sep='\t')

#this is giving the current median housing price in these regions

housing=housing[['period_begin','state','state_code','property_type','median_sale_price']]
housing=housing[(housing['period_begin']=='2022-01-01') & (housing['property_type']=='Single Family Residential')] 
housing.rename({'median_sale_price':'Median Sales Price ($)'},axis=1, inplace=True)

fig = px.choropleth(housing,
                    locations='state_code', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='Median Sales Price ($)',
                    color_continuous_scale="Viridis_r", 
                    
                    )

fig.show()


fig.update_layout(
      title_text = 'Jan 2022 Median Housing Price by State',
      title_font_family="Times New Roman",
      title_font_size = 22,
      title_font_color="black", 
      title_x=0.45,     
         )
