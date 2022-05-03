# ML-Climate-Final-Project-Template



Some notes on organization to make navigating this repsitory a little easier:
(All the notes below refer to source)

PRIMARY FILES SECTION

The file `main_prediction_yearly_agg` contains essentially the full pipeline, excepting the climate data projection data.
It is a bit unwieldy; I have tried to comment it well. It draws on historical storm data from the NWS Storm Events Database
and the GHCN (Global Historical Climatology Network) file for weather-related features, and culminates in training a model and making predictions.
The file `main_prediction_yearly_agg` contains similar code,(so please note--much of it may be repetitive), but instead of aggregating by year, it is altered so that it predicts per month. It is somewhat older, so it is less clean. I recommend reading main_prediction_yearly but it may still be of interest.

The file `storm_prediction_initial_clustering` contains some of my first work on the project, trying to detect patterns from clustering the data and generating a good plot. The paths may not even work at this point without some slight changes. I keep it here for the plots it can generate and as a demonstration of how much the project has changed.
The file useful_code contains, as the name suggests, a small list of utility functions that I found
to be useful at various stages of testing and experimenting.


READ AND PREPROCESS SECTION

The file `saving_files_locally` was just used for downloading the GHCN data from the web locally onto my computer. The code used for this is now commented out; The uncommented portion is the function I call to get the each feature via a dictionary with the feature name as the key and its associated dataframe as the value.

`load_and_process_storm_data` loads in the NWS historical storm events data from a list of files and combines them. It also does some basic pre-processing work to remove extraneous columns and convert the damage estimate to a proper float type. It saves an interim data file to csv which is later used in main_prediction.

The file `county_version_saving_files_locally` contains a lot of repetitive code from the file saving_files_locally;
I didn't have time to develop a clean file locally, and differences in how I need to process the ID made it
worth creating a differently file for. If you understand saving_files_locally, you understand this file.

The file `copernicus` contains code for calling the Climate Data Store API to download files of a given 
type. It also contains code for processing this data and beginning to map it onto a US map.

- The folder `unfinished_work` denotes code that I began but didn't havet time to finish, 
such as the GNN and the experiment logging. I hope to finish it some day and keep it as some of the most interesting ideas that I touched upon along the way during the process of iterating through things. Note on plotting housing data---it is mentioned in citations and in the file, much of the code was from an online article (not written by me). I have altered it but not enough to claim it as my code.


DICTIONARY SECTION

`state_to_neighbor_dict` is a dictionary mapping each state to its bordering states in list form.

`fips_code_dictionary` contains two dictionaries; one mapping each state to its fips code, and the other mapping the fips code to the state.

The file `us_state_to_abbrev` (not written by me) contains a simply utility function to get a dictionary of us state names 
to their two-letter abbreviation; again, I did not write this code myself. Citation in the file and in my report.

`temp_state_dict_working` contains a number of dictionaries for mapping the id in the GHCN/NWS data. Some of these are unique to the GHCN data, (i.e. they are not even standard FIPS codes--just taken from the GHCN website.)


A number of files have been removed and file names have been changed in the final commit to make this repository
more easily navigable and understood. Prior commits will contain more "scratch work" if it is of interest
to the reader.




Throughout the semester each student will work on an individual project, which you will summarize in a final technical paper. You will showcase and document your work through a private git repository following this template.

The organization of this repository is as follows:

```
abstract.md
journal.md
doc/
src/
etc/
```
- The file `abstract.md` simply contains an abstract of the project. At first, it is an aspirational abstract, one that describes the research program you want to complete. You will refine it through the semester.
- The file `journal.md` is a diary of your progress. It contains dated entries with a description of what you are doing, what you found, what you are thinking, and so on. It is mainly a resource for you, but I will glance at it too (at the end of the semester). Please update and commit it at least once per week.
- The `doc/` directory contains the LaTeX document that you are writing. We will provide a template for your final paper.
- The `src/` directory contains the code you are writing. The data you are analyzing should live here too.
- The `etc/` directory contains anything else — materials, notes, photos of whiteboards, and so on — that you want to keep track of.
There should be nothing else in the top level directory of your repository.

Commit often, at least every week. You are graded on the quality of the project and the path that you took to get there.