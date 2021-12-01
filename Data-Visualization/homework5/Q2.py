import json
from matplotlib import animation
from IPython.display import HTML
from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2.1 plot GDP
# # use country name as index column
df = pd.read_excel("GDP-fromworldbank.xls", skiprows=3, index_col=0)
countries = ['China', "United States", "Japan",
             "Singapore", "India", "United Kingdom"]
country_data = [df.loc[country][3:-1] for country in countries]
years = [y for y in range(1960, 2017)]
fig = plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
for i, country in enumerate(countries):
    x = years
    y = [t for t in country_data[i]]
    ax1.scatter(x, y, s=2, alpha=0.7,   label=country)
    ax2.plot(x, y, linewidth=1.0,  label=country)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    ax2.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))

plt.show()

# 2.2
fig = plt.figure(figsize=(10, 10))
map = Basemap(projection="cyl")
map.drawmapboundary()
map.fillcontinents(color='lightgray', zorder=1)
map.drawcoastlines()
map.drawcountries()


df = pd.read_excel("GDP-fromworldbank.xls", skiprows=3)
years = [str(y) for y in range(1960, 2017)]


with open("countries_latitude_longitude.json", 'r') as load_f:
    latlon_list = json.load(load_f)


def update(latency):
    print(years[latency])
    current_gdp = df[years[latency]]
    log_gdp = np.log10(current_gdp)
    rate = (log_gdp - np.min(log_gdp)) / \
        (np.max(log_gdp) - np.min(log_gdp))
    scat.set_sizes((1+rate)**8)
    year_legend.set_text(years[latency])
    scat.set_color(cmap(rate))
    # scat.set_color(log_gdp)
    return scat, year_legend,


def init():
    return scat, year_legend,


# map country code to latitude and longitude
name2lat = {}
name2long = {}
for item in latlon_list:
    name2lat[item["name"]] = item["latitude"]
    name2long[item["name"]] = item["longitude"]

df_loc = pd.DataFrame(columns=['lat', 'long'], data=[
                      [name2lat[name.strip()], name2long[name.strip()]] if name in name2lat and name in name2long else [None, None] for name in df["Country Name"]])

df = pd.concat([df, df_loc])

print(df['lat'])

# calculate GDP for each countries and year
MAX_GDP = np.max(df[years].fillna(0.00001))
MIN_GDP = np.min(df[years].fillna(0.00001))
current_gdp = df[years[0]].fillna(0.00001)
log_gdp = np.log10(current_gdp)
rate = (log_gdp - np.min(log_gdp)) / \
    (np.max(log_gdp) - np.min(log_gdp))

print(np.max(rate), np.min(rate))
print(rate)
cmap = plt.get_cmap('coolwarm')
year_legend = plt.text(-160, -70, str(years[0]), fontsize=15)
x, y = map(df['long'], df['lat'])
scat = map.scatter(x, y, (1+rate)**8, marker='o',
                   alpha=0.5, zorder=10,
                   cmap=cmap, c=rate)
cbar = map.colorbar(scat, location='bottom')
ani = animation.FuncAnimation(fig, update, interval=200, init_func=init,
                              frames=2017-1960, blit=True)

plt.title('Global GDP from 1960 to 2016')

ani.save("test.gif", writer='pillow')
plt.show()
