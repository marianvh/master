
from import_data import save, preprocessed_frame
from helpers import Helpers
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class TrajectoryCreate:
    
    def __init__(self):
        self.helper = Helpers()

    def create_trajectories(self, place):
        df = preprocessed_frame()
        df = df[["datetime","latitude","longitude", "place", "id", "Alt"]]
        if place is not None:
            df = df[df.place == place]
        save(f"unworked_frame_{place}", df)    

        sheep_trajectories = []
        sheep_ids = df['id'].unique()
        for id in sheep_ids:
            frame = df[df.id == id]
            frame = frame[["datetime","latitude", "longitude", "Alt", "id"]]
            trajectory = self.create_trajectory(frame)
            sheep_trajectories = [*sheep_trajectories, *trajectory]
        newf = pd.DataFrame(sheep_trajectories, columns=["latitude", "longitude", "distance", "days", "hours", "velocity", "rotation", "altitude", "id", "datetime"])
        
        save(f"{place}_traj",newf)



        #create one trajectory from data
    def create_trajectory(self, frame):
        latitude = np.array(np.deg2rad(frame.latitude),dtype=float)
        longitude = np.array(np.deg2rad(frame.longitude),dtype=float)
        times = np.array(frame.datetime,dtype="O")
        altitude = np.array(frame.Alt)
        ids = np.array(frame.id)
        trajectory_lines = np.c_[latitude[0:-1], longitude[0:-1],  latitude[1:], longitude[1:],  times[0:-1], times[1:], altitude[:-1], ids[:-1]]

        # format: latitude, longitude, days, distance, velocity, rotation, altitude and id
        #trajectory = list(map(lambda linesegment: [distvelrot(*linesegment[:6]), linesegment[6], linesegment[7], linesegment[4]], trajectory_lines) )
        trajectory = []
        for row in trajectory_lines:
            dist, days, hours, vel, rot = self.helper.distvelrot(*row[:6])
            #print(row[0],row[1], dist, time,vel, rot, row[6], row[7], row[4])
            trajectory.append(np.array([np.rad2deg(row[0]),np.rad2deg(row[1]), dist, days, hours, vel, rot, row[6], row[7], row[4]]))
        #trajectory = list(map(lambda linesegment: distvelrot(*linesegment[:,:6]), trajectory_lines) )

        return trajectory
