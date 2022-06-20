# master
DBSCAN of sheep GPS data

This repo is structured pretty bad as it was urgent, but:

_altitude_analysis.py_ plots various graphs based on altitude
_cleanSheep.py_ cleans the dataframe
_cluster.py_ is the class that runs dbscan
_distance_per_hour_of_day.py_ also contains altitude plotters
_filtering.py_ is a failed implementation of different smoothing filters
_helpers.py_ is the class that contains code for creating trajectories, calculating values and distance matrixes, and contains the custom distance function
_main.py_ used for testing and plotitng
_plotdata.py_ various functions that plot data in different ways
_plotter.py_ sort of interface for plotting. Was abandoned
_preprocess_clustering_ ran DBSCAN on the data to remove outliers
_trajclus_ creates trajectories using _helpers_ 
