# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:13:12 2022

@author: 16028
"""

'''
Hi-- longer post to explain my changes. I have really been struggling to get excited
about my initial project idea. After lots of indecision and consideration, I've decided
to switch to a new project idea that I am really excited about. The main idea
is the same--predicting a hazardous weather event that is likely to be influenced by climate change, using historical data, 
and doing some extrapolation into the future by merging it with climate data (and future projections). I want
to develop a model to predict how the severity and distribution of storms in the US in different regions will change over 
time. 
This would be done using an autoencoder to develop some kind of model of the dynamics between severe storms
and climate conditions in the US ,as well as another machine learning model layered 
on top to use the dynamics as features (right now I'm thinking more deep learning--a CNN--because of the nature of the data). 
                                        
I also want to do some clustering of zip codes in the US by cost of storm 
damage (included in the dataset) to produce a 'climate vulnerability' ranking system. The techniques I want to examine are K-Means,
Spectral Clustering, and Gaussian Mixtures. This ranking system could be used by actual 
homeowner insurance companies and businesses deciding on regions to expand into.
The dataset I am using is from the NOAA Severe Storm Events Database--(see 'https://www.ncdc.noaa.gov/stormevents/' 
                                                                       'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'). 
This storm events in the data are also accompanied by a narrative text description that I would really like to use as a feature in the model
but that I will only get to if time permits.
I am intrigued by the idea of using'climate insurance risk' modeling to effect
changes in climate policy--if we can better price in the climate risk of real estate properties in certain zip codes
in the US, we will be both more climate resilient and we can help develop some smarter market incentives to reduce carbon emissions.
I know I'm behind so I will be pushing updates more frequently.

If I'm not permitted to switch topics, please let me know now! Otherwise I am looking forward to working on this and pushing more updates soon.

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

Need to do: 
1.Finish settting up GCP to use more compute power
2. Merge climate conditions data with this dataset
3. Use Google Notebook tutorial to merge lat/long attirbutes with geographic zip code data
3. Convert latitude and longitude data to grid format for input to CNN
4. Develop CNN to see how it performs on the sparse geographic data.
5. Develop some better visualizations of the data for initial exploration.
6. Fix issue with altair package install not working.
'''