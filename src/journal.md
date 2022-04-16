Week 1-2:

Wrote short script to combine the collection of datasets offered by sierra nevada mountains. 
Having some strange data cleaning issue that's making it difficult to make even basic 
exploratory plots. Continuing to work on fixing this to make the plots work.

Week 3 ish:
Current bug: can't open the NOAA file because it is .nc and I don't have a compatible library
(yet) for that data type:
Current issue: import line not working
next to try: try conda uninstalling netCDF4 and then using pip install netcdf4?


 Read the following papers:
1. https://essd.copernicus.org/preprints/essd-2021-399/ provides ideas for features
2. https://journals.ametsoc.org/view/journals/bams/aop/BAMS-D-20-0243.1/BAMS-D-20-0243.1.xml --crucially provides dataset
3. https://www.pnas.org/content/117/33/19753 --good overview, background


Identified the following datasets as candidates in addition to the Central Sierra Mountains
(still deciding on whether to add datasets to my original problem setup to see if that 
improves that generalizability of my analysis):
1. https://www.weather.gov/wrh/Climate?wfo=ind
2. https://nsidc.org/data/g02158 has a dataset that I want to pull but there's some difficulty
getting it--can't just downlaod from web, need to write a python script to pull it. Currently
working on script.
3. this repository as well: https://github.com/alex-gottlieb/snow_drought --need to download the .yml file to access the data?


Readings to consider doing next week:
1. https://journals.ametsoc.org/view/journals/hydr/14/3/jhm-d-12-0119_1.xml for connection to water budgets/water security in colorado basin


#####################################################################################################
MAJOR CHANGE
#####################################################################################################

Hi-- longer post to explain my changes. I have really been struggling to get excited 
about my initial project idea. After lots of indecision and consideration, I've decided
to switch to a somewhat different project that I am really excited about. The main idea is
the same--predicting a hazardous weather event that is likely to be influenced by
climate change, using historical data, and doing some extrapolation into the future
 by merging it with climate data (and future projections). I want to develop a 
 model to predict how the severity and distribution of storms in the US in different
regions will change over time. This would be done using an autoencoder todevelop 
some kind of model of the dynamics between severe storms in the US and climate conditions, 
as well as another machine learning model layered on top to use the dynamics as 
features (right now I'm thinking more deep learning--a CNN--because of the nature of the data). 
I also want to do some clustering of zip codes in the US by cost of storm 
damage (included in the dataset) to produce a 'climate vulnerability' ranking system.
The techniques I want to examine are K-Means, Spectral Clustering, and Gaussian 
Mixtures. This ranking system could be used by actual homeowner insurance companies 
and businesses deciding on regions to expand into. The dataset I am using is from the
 NOAA Severe Storm Events Database--(see 'https://www.ncdc.noaa.gov/stormevents/'and
'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'). 
This storm events in the data are also accompanied by a narrative text description 
that I would really like to use as a feature in the model but that I will only 
get to if time permits.I am intrigued by the idea of using'climate insurance risk' modeling to effect
changes in climate policy--if we can better price in the climate risk of real estate 
properties in certain zip codes in the US, we will be both more climate resilient 
and we can help develop some smarter market incentives to reduce carbon emissions. 

If I'm not permitted to switch topics, please let me know now!I know I am 
behind at this point so I will be pushing updates more frequently. Otherwise I am looking
forward to working on this and pushing more updates soon.
'''

'''
Week 2/28: 
Found dataset, extracted and read in data. 
Found Google public notebook giving a tutorial on combining storm data
with geographic zip code and location (this notebook also gave me the main idea of using storm data for insurance modeling--I'll cite the code when I integrate it in)
Wrote script to read in all the data files in the directory.
Started process of settting up GCP account using credits.
Did some very basic clustering 
Develop a a simple random forest model just to get a sense of the data.

Need to do next: 
1.Finish settting up GCP to use more compute power
2. Merge climate conditions data with this dataset
3. Use Google Notebook tutorial to merge lat/long attirbutes with geographic zip code data
3. Convert latitude and longitude data to grid format for input to CNN
4. Develop CNN to see how it performs on the sparse geographic data.
5. Develop some better visualizations of the data for initial exploration.
6. Fix issue with altair package install not working.
'''

2022-03-07 check in: alp

Looks good. Would recommend getting some initial, even partial results, and to continue the data merging in parallel. Insights from modeling the small, potentially unremarkable data could help with the data wrangling and inspire new ideas.
