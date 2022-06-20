from sklearn.preprocessing import MinMaxScaler
from import_data import preprocessed_frame, load
import matplotlib.pyplot as plt
import seaborn as sns
q = load("preprocessed_frame")
df = preprocessed_frame()

def plot(df):
    data = df
    ids = data.id.unique()
    yrs = data.datetime.dt.year.unique()
    data["hour"] =  data.datetime.dt.hour*2
    data["hour"] =  data.hour.where(data.datetime.dt.minute < 30, data.hour+1) 
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    for ddd in ids:
        d = data[data.id == ddd]
        ilist = data[data.id == ddd].index.tolist()
        data.loc[ilist,"mov_avg_alt"] = d.Alt.rolling(200).mean()
        
    for year in yrs:
        d = data[data.datetime.dt.year == year]
        sns.lineplot(
                x="hour",
                y="Alt",
                data= d,
                label=year
        )   
    plt.xlabel("Time of Day")
    plt.ylabel("Altitude")



    plt.show()
    

def plot_coordinates_hours(df, place, year, dawninterval=5, duskinterval=19):
    data = df[df.place == place]
    data["hour"] = data.datetime.dt.hour
    data = data[data.datetime.dt.year == year]
    ids = data.id.unique()
    
    scaler = MinMaxScaler()
    data[["latitude", "longitude"]] = scaler.fit_transform(data[["latitude", "longitude"]])
    nighttime = data[(data.datetime.dt.hour >= duskinterval) | (data.datetime.dt.hour <= dawninterval)]
    daytime = data[(data.datetime.dt.hour < duskinterval) & (data.datetime.dt.hour > dawninterval)]
    
    sns.scatterplot(
        x="longitude",
        y="latitude",
        data= daytime,
        alpha=0.9
    )   
    sns.scatterplot(
        x="longitude",
        y="latitude",
        data= nighttime,
        alpha=0.9
    )   
    plt.show()
