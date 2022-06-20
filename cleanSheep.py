import numpy as np
import pandas as pd
from import_data import fetch_sheep_data, fetch_sheep_info
from scipy import stats


# Cleans dataframe and saves as pkl
class CleanSheep:
    def __init__(self, placename, sheep_data, sheep_info):
        sheep_data = self._clean_sheep_data(sheep_data, placename)
        sheep_data = self._clean_using_sheep_info(sheep_info, sheep_data)
        self.sheep_data = sheep_data
        
    def _clean_sheep_data(self, sheep_data, placename):    

        # remove features we don't want
        def general_clean(inputframe):
            frame = inputframe.drop(["Nan0", "Status", "SCap", "GPS", "GSM", "Status", "NaN2", "date", "time"], axis=1)
            # The latitude and longitude columns contained status strings
            frame["TimeOut"] = np.where(frame.latitude == "Time Out", True, False)
            frame["latitude"] = np.where(frame.latitude == "Time Out", np.nan, frame.latitude)
            frame["Mortality"] = np.where(frame.latitude == "Mortality", True, False)
            frame["latitude"] = np.where(frame.latitude == "Mortality", np.nan, frame.latitude)
            frame["GPSerror"] = np.where(frame.latitude == "GPS Error", True, False)
            frame["latitude"] = np.where(frame.latitude == "GPS Error", np.nan, frame.latitude)
            frame = frame.dropna(subset=['latitude', 'longitude'])
            # straight up remove data points from before grazing season
            frame = frame[frame.datetime.dt.month >= 6]
            frame.latitude = frame.latitude.astype(float)
            frame.longitude = frame.longitude.astype(float)

            frame['Alt'] = frame['Alt'].where(frame['Alt'] >10, frame['Alt'].shift(-1))
            frame['Alt'] = frame['Alt'].where(frame['Alt'] >10, frame['Alt'].shift(-1))
            return frame

        def remove_geographical_outliers(data):
            assert "latitude" in data.columns, "column name not in data set"
            assert "longitude" in data.columns, "column name not in data set"
            data = data[(np.abs(stats.zscore(data.latitude)) < 1)]
            data = data[(np.abs(stats.zscore(data.longitude)) < 2)]
            return data
        
        def none_outlier_years(data):
            #remove years that should not be in the dataset
            assert "datetime" in data.columns, "column name not in data set"
            d = dict(data.datetime.dt.year.value_counts())
            years = []
            for key, value in d.items():
                if(value > 1000):
                    years.append(key)
            return years
            
        frame = sheep_data[sheep_data['datetime'].notnull()].copy()
        frame = general_clean(frame)
        frame = frame[frame.datetime.dt.year.isin(none_outlier_years(frame))] 
        frame = remove_geographical_outliers(frame)
        frame = self._remove_start_and_end_dates(frame)
        # or kalman or median :)
        #frame = mean_filter(frame)
        return frame
    
    def _remove_start_and_end_dates(self, sheepData):
        ids = list(sheepData.id.unique())
        df = sheepData.copy()
        for id in ids:
            newframe = sheepData[sheepData.id == id]
            startdate = newframe.iat[0,18].date()
            enddate = newframe.iat[-1,18].date()
            df = df[df.datetime.dt.date != startdate ]
            df = df[df.datetime.dt.date != enddate ]
            
        return df
        
    def _clean_using_sheep_info(self, sheepInfo, sheepData):
        # remove data points logged same day the sheep were let out to graze
        # remove data points logged the last day the tracker moved
        df = pd.merge(sheepData, sheepInfo, left_on="id", right_on="combined_id", how="left", indicator=True)

        df = df[df.slippdato.dt.date != df.datetime.dt.date]
        df = df.drop(sheepInfo.columns, axis=1)
        df = df.drop(["_merge"], axis=1)
        return df

    def return_dataset(self):
        return self.sheep_data
    
    def head(self):
        print(self.sheep_data.head())

    def save_data(self):
        self.sheep_data.to_pickle("combined_df.pkl")

        