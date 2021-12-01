import json
from matplotlib import animation
from IPython.display import HTML
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

quakes = pd.read_csv("quakes.csv")

fig = plt.figure(figsize=(10, 10))


map = Basemap(projection='merc', resolution='l', area_thresh=1200.0,
              lat_0=0, lon_0=130,
              llcrnrlon=min(quakes.long), llcrnrlat=min(quakes.lat),
              urcrnrlon=max(quakes.long), urcrnrlat=max(quakes.lat))

plt.title("Earthquakes Location")

map.drawcoastlines()
map.drawcountries()

map.drawmapboundary(fill_color='powderblue')
map.fillcontinents(color='#ddaa66', lake_color='aqua')
max_depth = max(quakes.depth)
for i, item in quakes.iterrows():
    x, y = map(item.long, item.lat)
    # 点越大，表示地震强度越大
    makersize = item.mag ** 3
    # 颜色越深, 表示震源越深
    colors = np.array([0.2, 0.0, item.depth/(max_depth+100)])
    map.scatter(x, y, makersize, color=colors, marker='o',
                alpha=0.5)

plt.show()
