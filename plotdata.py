import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
import numpy as np


def scatterplot(data):
    plt.scatter(data.longitude, data.latitude, c="blue",alpha=0.4)
    plt.show()

def trajectory_plot(data):
    plt.plot(data[:, 0], data[:, 1])
    plt.show()  

def partitionplot(lat, lon,labels):
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    labs = list(map(lambda label: colors[label], labels))
    plt.scatter(lat, lon, color=labs)
    plt.show()

def trajectorycolorplot(data, label):
    x = data[:,1]
    y = data[:,0]
    c = np.arange(0,1,1/len(data))
    lines = np.c_[x[:-1], y[:-1], x[1:], y[1:]]
    lc = LineCollection(lines.reshape(-1, 2, 2), array=c, linewidths=1)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    bar = fig.colorbar(lc)
    bar.set_label('Tracking time, start is dark')
    ax.autoscale()
    plt.title(label)
    plt.xlabel("longitude normalised")
    plt.ylabel("latitude normalised")
    plt.show()

    
def clusterplot(labels, data, core_samples_mask):
    unique_labels = set(labels)
    
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    #plott alle punkter i samme farge!
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        core_samples_mask.astype(bool)
        xy = data[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 1],
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 1],
            xy[:, 0],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(len(data))
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()    
