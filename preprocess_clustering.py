from import_data import process_combined_frame
from helpers import distance_self
from sklearn.cluster import DBSCAN
from plotdata import scatterplot
import numpy as np
from import_data import save, load



class Preprocess(): 
    def __init__(self):
        df = process_combined_frame()
        df = df[["datetime","latitude","longitude", "place", "id", "Alt"]]
        self.df = df
        self.ids = df.id.unique()
             

    def dbscan_(self, startidnumber, cluster=False, saved=False):
        ids = self.ids[startidnumber:startidnumber+3]
        print(ids)
        data = self.df[self.df.id.isin(ids)]



        if(cluster or saved):
            distance_matrix = distance_self(np.array(data[["latitude", "longitude"]]))
            clusterer = DBSCAN(eps=150, min_samples=7, metric='precomputed', metric_params=None, algorithm='auto',  p=None, n_jobs=None).fit(distance_matrix)
            labels = clusterer.labels_
            unique_labels = list(set(labels))
            data["labels"] = labels
            df1 = data[data.labels.isin(unique_labels[0:1])]

            if(cluster):
                scatterplot(df1)
            if(saved):
                f = load("preprocessed_frame")
                print(f.head())
                f = f.append(df1)
                save("preprocessed_frame",f)
        else:
            scatterplot(data[["latitude", "longitude"]])

