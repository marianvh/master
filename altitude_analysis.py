from import_data import preprocessed_frame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

df = preprocessed_frame()
def height_graph(frame, year, place, rolling_window=100):
    frame = frame[frame.datetime.dt.year == year]
    frame = frame[frame.place == place]
    frame['mov_avg'] = frame['Alt'].rolling(rolling_window).mean()
    print(frame.mov_avg.head(20))
    print(frame.Alt.head(10))
    plot(frame)

def plot(frame):
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    ids = frame.id.unique()
    for id in ids:
        sns.lineplot(
            x='datetime',
            y='mov_avg',
            data= frame[frame.id == id])
    plt.show()

def check_num_of_dates(df, place, year=None):
    df = df[df.place == place]
    if(year is not None):
        df =df[df.datetime.dt.year == year]

    years = df.datetime.dt.year.unique()
    df["date"] = df.datetime.dt.date # df.datetime.dt.strftime('%m-%d')
    

    for yr in years:
        newf = df[df.datetime.dt.year == yr]
        newf['mov_avg'] = df['Alt'].rolling(100).mean()
        newf['mov_avg'] = newf['mov_avg'].fillna(0)
        fig, ax = plt.subplots(figsize=(14, 9))
        sns.lineplot(
            x = 'date',
            y = 'mov_avg',
            data=newf
        )
        
        item_counts = dict(newf["date"].value_counts())
        counts = np.array(list(item_counts.values())).T
        dates = np.array(list(map(lambda key: key.replace(year=year),item_counts.keys()))).T
        combined = np.c_[dates, counts]
        print(combined[:4])
        sorted(combined, key=lambda row: row[0], reverse=True)
        data = pd.DataFrame(combined, columns=["dates", "counts"])
        
        sns.lineplot(
            x='dates',
            y='counts',
            data= data)
    plt.show()