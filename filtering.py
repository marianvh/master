from helpers import haversine_distance
import pandas as pd

#this was created before helpers became a class. Does not work. 

# Wrong implementation, does not work 
def mean_filter(frame, max_distance=3, n_size= 12):
    meanlat, meanlong = 0,0
    frame = frame.reset_index()
    datalength = len(frame)
    window_start = 0
    window_stop = n_size
    newframe = frame.copy()

    while window_stop < datalength:
        dataset_window = frame.loc[:,["latitude", "longitude"]]
        # don't include own coordinate
        dataset_window = dataset_window.iloc[window_start:window_stop-1]
        meanlat = dataset_window.latitude.mean()
        meanlong = dataset_window.longitude.mean()
        lat = frame.at[window_stop,'latitude']
        long= frame.at[window_stop,'longitude']
        # if distance between new point is max distance away from average :)
        if(abs(haversine_distance(meanlong, meanlat, long,lat )) > max_distance):
            newframe.at[window_stop,'latitude'] = meanlat
            newframe.at[window_stop,'longitude'] = meanlong
        window_start += 1
        window_stop += 1
    return newframe
    

def median_filter(frame, max_distance=3, n_size= 12):
    medianlat, medianlong = 0,0
    frame = frame.reset_index()
    # gå gjennom alle punktene, hvis de er en viss error vekk fra mean så bytt ut med mean??
    datalength = len(frame)
    window_start = 0
    window_stop = n_size
    newframe = frame.copy()

    while window_stop < datalength:
        dataset_window = frame.loc[:,["latitude", "longitude"]]
        # don't include own coordinate
        dataset_window = dataset_window.iloc[window_start:window_stop-1]
        medianlat = dataset_window.latitude.median()
        medianlong = dataset_window.longitude.median()
        lat = frame.at[window_stop,'latitude']
        long= frame.at[window_stop,'longitude']
        if(abs(haversine(medianlong, medianlat, long,lat )) > max_distance):
            newframe.at[window_stop,'latitude'] = medianlat
            newframe.at[window_stop,'longitude'] = medianlong
        window_start += 1
        window_stop += 1
    return newframe
        
def kalman_filter():
    return
    # https://nbviewer.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb
               