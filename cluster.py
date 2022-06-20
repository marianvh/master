import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from helpers import Helpers
from import_data import save, load
from plotdata import clusterplot
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time


class Cluster:
    def __init__(self, place,num_of_sheep=1,start_sheep=1,changed=False):
        self.start_time = time.time()

        self.sheep_trajectories = load(f"{place}_traj")
        df = load(f"unworked_frame_{place}")
        self.df = df
        self.helper = Helpers(trajectories=self.sheep_trajectories,num_of_sheep=num_of_sheep, start_sheep=start_sheep, changed=changed)
        self.num_of_sheep = num_of_sheep
        self.start_sheep = start_sheep
        return
    
    def kmeans_(self, num_of_sheep=1, start_sheep=0, n_clusters=7):
        print(self.sheep_trajectories.head())
        ids = self.sheep_trajectories.id.unique()
        newids = []
        # the following steps are confusing, but it's to get the same order as the data set is read
        for i in ids:
            newids.append(str(self.sheep_trajectories[self.sheep_trajectories.id == i].iloc[0].datetime.year)+"_"+i)
        newids.sort()
        scaler = MinMaxScaler()
        sheep_to_use = list(map(lambda i: i.split('_',1)[1],newids))[start_sheep:start_sheep+num_of_sheep]
        data = self.sheep_trajectories[self.sheep_trajectories.id.isin(sheep_to_use)]
        #data[["distance", "days", "velocity", "rotation", "altitude"]] = scaler.fit_transform(data[["distance", "days", "velocity", "rotation", "altitude"]])

        data[['latitude', 'longitude']] = np.deg2rad(data[['latitude', 'longitude']])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        label = kmeans.fit_predict(data[["latitude", "longitude"]])
        
        u_labels = np.unique(label)
        print(u_labels)
       
        data[["latitude", "longitude"]] = scaler.fit_transform(data[["latitude", "longitude"]])
        data = np.array(data)
        #plotting the results:
        for i in u_labels:
            plt.scatter(data[label == i , 1] , data[label == i , 0] , label = i)
        plt.legend()
        plt.show()

    def fetch_part_of_trail(self,stop=100, start=0, year=2012):
        data = self.df[self.df.datetime.dt.year == year]
        sheep_ids = data['id'].unique()
        temp = []
        plotframe = []
        mean_lat = self.df.latitude.mean()
        mean_long = self.df.longitude.mean()
        std_lat = self.df.latitude.std()
        std_long = self.df.longitude.std()
        for id in sheep_ids:
            slicey = data[data.id == id]
            slicey = slicey[["datetime", "latitude", "longitude"]]
            temp = [*temp, *np.array(slicey.iloc[start:stop])]

            slicey["latitude"] = (self.df.latitude - mean_lat)/std_lat
            slicey["longitude"] = (self.df.longitude - mean_long)/std_long

            plotframe = [*plotframe, *np.array(slicey[["latitude", "longitude"]].iloc[start:stop])]
        
        temp = np.array(temp)
        plotframe = np.array(plotframe)[:-1]
        traj = self._create_trajectory(temp)
        dist = self.helper.distance_self(np.array(traj))
        self._dbscan_partition(dist, plotframe, True)   

    def _dbscan_partition(self, trajectory, plotframe, distance_matrix=False):
        if distance_matrix:
            clusterer = DBSCAN(eps=90, min_samples=10, metric="precomputed", metric_params=None, n_jobs=None).fit(trajectory)
        else: 
            clusterer = DBSCAN(eps=0.08, min_samples=10, metric="euclidean", metric_params=None, n_jobs=None).fit(trajectory)
        core_samples_mask = np.zeros_like(clusterer.labels_, dtype=bool)
        core_samples_mask[clusterer.core_sample_indices_] = True
        labels = clusterer.labels_

       
        save("result", np.c_[labels, plotframe, core_samples_mask])
        clusterplot( labels, plotframe, core_samples_mask)

    


    def dbscan_(self,eps=0.02, min_samples=30,columns=["latitude", "longitude", "distance", "days", "velocity", "rotation", "altitude"], precomputed=False):        
        if self.num_of_sheep is None:
            data = self.sheep_trajectories
        else:
            ids = self.sheep_trajectories.id.unique()
            newids = []
            for i in ids:
                newids.append(str(self.sheep_trajectories[self.sheep_trajectories.id == i].iloc[0].datetime.year)+"_"+i)
            newids.sort()
            sheep_to_use = list(map(lambda i: i.split('_',1)[1],newids))[self.start_sheep:self.start_sheep+self.num_of_sheep]
            data = self.sheep_trajectories[self.sheep_trajectories.id.isin(sheep_to_use)]
        scaler = QuantileTransformer()
        data["index"] = np.arange(0, len(data))
        data["rotation"] = np.rad2deg(data.rotation)
        data[["distance", "velocity", "rotation", "altitude", "days"]] = scaler.fit_transform(data[["distance", "velocity", "rotation", "altitude", "days"]])

        if(precomputed):
            clusterer = DBSCAN(eps=eps,min_samples=min_samples, metric=self.helper.custom_metric, metric_params=None, algorithm='ball_tree',  p=None, n_jobs=None ).fit(data[columns])
        else:
            clusterer = DBSCAN(eps=eps,min_samples=min_samples, metric=self.helper.custom_metric, algorithm="brute",metric_params=None, p=None, n_jobs=None ).fit(np.array(data[columns]))
        labels = clusterer.labels_

        data["labels"] = labels
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[clusterer.core_sample_indices_] = True
        data["core_samples_mask"] = core_samples_mask
        save("result", data)
        scaler1 = MinMaxScaler()
        data[["latitude", "longitude"]] = scaler1.fit_transform(data[["latitude", "longitude"]])
        data = np.array(data[["latitude", "longitude"]])
        
        clusterplot( labels, data, core_samples_mask)
       