# master
DBSCAN of sheep GPS data

This repo is structured pretty bad as it was urgent, but:

_altitude_analysis.py_ plots various graphs based on altitude <br/>
_cleanSheep.py_ cleans the dataframe <br/>
_cluster.py_ is the class that runs dbscan <br/>
_distance_per_hour_of_day.py_ also contains altitude plotters <br/>
_filtering.py_ is a failed implementation of different smoothing filters <br/>
_helpers.py_ is the class that contains code for creating trajectories, calculating values and distance matrixes, and contains the custom distance function <br/>
_main.py_ used for testing and plotitng <br/>
_plotdata.py_ various functions that plot data in different ways <br/>
_plotter.py_ sort of interface for plotting. Was abandoned <br/>
_preprocess_clustering_ ran DBSCAN on the data to remove outliers <br/>
_trajclus_ creates trajectories using _helpers_  <br/>
