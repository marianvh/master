import numpy as np
from plotdata import scatterplot, trajectorycolorplot, trajectory_plot
from sklearn.preprocessing import MinMaxScaler


def plot_trajectory(df, type: str, ids: str, traj_start: int, traj_length:int = None) -> None:
    data = df[df.id == ids]
    year = int(df.datetime.dt.year.median())
    data = data[["latitude", "longitude"]]
    scaler = MinMaxScaler()
    data[["latitude", "longitude"]] = scaler.fit_transform(data[["latitude", "longitude"]])
    
    if traj_length is not None: 
        data = data.iloc[traj_start:traj_start+traj_length]

    if(type=="c"):
        trajectorycolorplot(np.array(data), f'Sheepid {ids} year {year}')
    else:
        trajectory_plot(data.to_numpy())
  
    
def plot_year(df, place:str, year:int=2012):
    data = df[df.place == place]
    data = data[data.datetime.dt.year == year]
    scatterplot(data)


def plot_all_years(df):
    years = df.datetime.dt.year.unique()
    for year in years:
        plot_year(df, year)
            
def plot_single_sheep(df, sheepnr, year):
    data = df[df.datetime.dt.year == year]
    ids = list(data.id.unique())[sheepnr]
    data = data[data.id == ids]
        
    scatterplot(data)

