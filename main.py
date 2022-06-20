from import_data import process_combined_frame, load
from sklearn.preprocessing import MinMaxScaler
from plotdata import clusterplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta



df =load("preprocessed_frame")
q = process_combined_frame()
clustered_frame = load("result")

def avgdatetime(yourtimedeltalist):
    times=pd.to_datetime(pd.Series(yourtimedeltalist))
    print(times.mean())
    print(times.min())
    print(times.max())


def check_num_of_dates(df, place):

    years = df.datetime.dt.year.unique()
    df = df[df.place == place]
    df["date"] = df.datetime.dt.date # df.datetime.dt.strftime('%m-%d')

    fig, ax = plt.subplots(figsize=(14, 9))

    for year in years:
        newf = df[df.datetime.dt.year == year]
        item_counts = dict(newf["date"].value_counts())
        counts = np.array(list(item_counts.values())).T
        dates = np.array(list(map(lambda key: key.replace(year=2012),item_counts.keys()))).T
        combined = np.c_[dates, counts]
        print(combined[:4])
        sorted(combined, key=lambda row: row[0], reverse=True)
        data = pd.DataFrame(combined, columns=["dates", "counts"])
        sns.lineplot(
            x='dates',
            y='counts',
            data= data)
    plt.show()

def altitude_per_cluster():
    for label in clustered_frame.labels.unique():
        sns.histplot(
                x='hours',
                y='altitude',
                data=clustered_frame[clustered_frame.labels == label])
    plt.show()

def consecutive_chain(frame):
    frame["sequences"] = False
    frame["sequences"] = frame.datetime + timedelta(minutes=30) == frame.datetime.shift(-1)
    frame["ends"] = "start"
    frame["ends"] = frame.sequences.mask((frame.sequences == True) & (frame.sequences.shift(1)==False), "start")
    frame["ends"] = frame.ends.mask((frame.sequences == False) & (frame.sequences.shift(1)==True), "end")
    return frame


def tolk_result(df, place):
    traj = load(f"{place}_traj")
    scaler = MinMaxScaler()
    traj[["latitude", "longitude"]] = scaler.fit_transform(traj[["latitude", "longitude"]])
    traj = traj[traj.id.isin(df.id.unique())]
    traj["labels"] = df.labels
    traj = traj[traj.labels == -1]
    consecutives = consecutive_chain(traj)
    seq = consecutives[consecutives.ends.isin(["end", "start", True])]
    seq.iloc[0,12] = "start"
    sns.scatterplot(x="longitude", y="latitude", data=traj, alpha=0.4)
    sns.scatterplot(x="longitude", y="latitude", data=seq, color="black")
    plt.show()
