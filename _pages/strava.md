---
permalink: /projects/strava/
title: "Visualizing Eight Years of Strava Activity"
classes: wide
author_profile: true
---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
from matplotlib.lines import Line2D
```

### Import and view data


```python
df = pd.read_csv ('~/Desktop/Code/StravaData/activities.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Activity ID</th>
      <th>Activity Date</th>
      <th>Activity Name</th>
      <th>Activity Type</th>
      <th>Activity Description</th>
      <th>Elapsed Time</th>
      <th>Distance</th>
      <th>Relative Effort</th>
      <th>Commute</th>
      <th>Activity Gear</th>
      <th>...</th>
      <th>Cloud Cover</th>
      <th>Weather Visibility</th>
      <th>UV Index</th>
      <th>Weather Ozone</th>
      <th>&lt;span class="translation_missing" title="translation missing: en-US.lib.export.portability_exporter.activities.horton_values.jump_count"&gt;Jump Count&lt;/span&gt;</th>
      <th>&lt;span class="translation_missing" title="translation missing: en-US.lib.export.portability_exporter.activities.horton_values.total_grit"&gt;Total Grit&lt;/span&gt;</th>
      <th>&lt;span class="translation_missing" title="translation missing: en-US.lib.export.portability_exporter.activities.horton_values.avg_flow"&gt;Avg Flow&lt;/span&gt;</th>
      <th>&lt;span class="translation_missing" title="translation missing: en-US.lib.export.portability_exporter.activities.horton_values.flagged"&gt;Flagged&lt;/span&gt;</th>
      <th>&lt;span class="translation_missing" title="translation missing: en-US.lib.export.portability_exporter.activities.horton_values.avg_elapsed_speed"&gt;Avg Elapsed Speed&lt;/span&gt;</th>
      <th>&lt;span class="translation_missing" title="translation missing: en-US.lib.export.portability_exporter.activities.horton_values.dirt_distance"&gt;Dirt Distance&lt;/span&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>159187244</td>
      <td>Jun 28, 2014, 4:48:31 PM</td>
      <td>Horsetooth</td>
      <td>Ride</td>
      <td>NaN</td>
      <td>4641</td>
      <td>14.67</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>159314142</td>
      <td>Jun 28, 2014, 10:41:53 PM</td>
      <td>Antelope trail</td>
      <td>Ride</td>
      <td>NaN</td>
      <td>5455</td>
      <td>8.79</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>160874659</td>
      <td>Jul 2, 2014, 4:07:30 PM</td>
      <td>North Fruita Desert</td>
      <td>Ride</td>
      <td>NaN</td>
      <td>7846</td>
      <td>21.89</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>160971473</td>
      <td>Jul 2, 2014, 8:00:21 PM</td>
      <td>Horsethief loop</td>
      <td>Ride</td>
      <td>NaN</td>
      <td>5257</td>
      <td>13.34</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>161316446</td>
      <td>Jul 3, 2014, 3:48:59 PM</td>
      <td>The Whole Enchilada (before phone died)</td>
      <td>Ride</td>
      <td>NaN</td>
      <td>10706</td>
      <td>16.55</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>



### Format activity date column as datetime object


```python
df['Activity Date'] = list(map(lambda x: datetime.strptime(x,'%b %d, %Y, %H:%M:%S %p'),df['Activity Date']))
df['Year'] = list(map(lambda x: x.year, df['Activity Date']))
years = np.unique(np.array(df['Year']))
array = np.zeros((years.size,7,53))
```

### Create activity arrays for visualization


```python
yearcount = 0
for i in range(1,df['Activity Date'].size):
    lastdate = df['Activity Date'][i-1]
    date = df['Activity Date'][i]
    x = (date.weekday(),date.week,date.year)
    if date.year != lastdate.year:
        yearcount+=1
    if df['Activity Type'][i] == 'Run':
        array[yearcount,date.weekday(),date.week%53] = 1
    if df['Activity Type'][i] == 'Ride':
        array[yearcount,date.weekday(),date.week%53] = 2
    if df['Activity Type'][i] == 'Nordic Ski':
        array[yearcount,date.weekday(),date.week%53] = 3
    if df['Activity Type'][i] == 'Hike':
        array[yearcount,date.weekday(),date.week%53] = 4
    if df['Activity Type'][i] == 'Alpine Ski':
        array[yearcount,date.weekday(),date.week%53] = 5
    if df['Activity Type'][i] == 'Canoe':
        array[yearcount,date.weekday(),date.week%53] = 6
```

### Define visualization colors


```python
none = 'black'
run = 'maroon'
ride = 'darkorange'
nordicski = 'darkcyan'
hike = 'yellow'
alpineski = 'azure'
canoe = 'midnightblue'
```

### Create visualization


```python
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(8,figsize=(24,28))
plt.subplots_adjust(hspace=0.1)
for j in range(0,years.size):
    if j == years.size-1:
        axs[j].set_xlabel('Week of the Year',fontsize=26, color = 'w')
    if j == 0:
        axs[j].set_title('Strava Activities',fontsize=26, color = 'w')
    axs[j].set_facecolor('black')
    axs[j].grid(False)
    axs[j].set_ylabel(f"{years[j]}",fontsize=26, color = 'w')
    fig.patch.set_facecolor('black')
    activities = array[j]
    X,Y = np.meshgrid(np.arange(activities.shape[1]), np.arange(activities.shape[0]))
    colors = {0.0:none,1.0:run, 2.0:ride, 3.0:nordicski,
              4.0:hike, 5.0:alpineski, 6.0:canoe}
    axs[j].scatter(X.flatten(), abs(Y.flatten()-6), c=pd.Series(activities.flatten()).map(colors), s = 500)
    axs[j].set_xlim(-1,53)
    axs[j].set_ylim(-1,7)
    axs[j].set_yticks(ticks = [6,5,4,3,2,1,0])
    axs[j].set_yticklabels(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                           fontsize=20, color = 'w')
    axs[j].spines['top'].set_visible(False)
    axs[j].spines['right'].set_visible(False)
    axs[j].spines['bottom'].set_visible(False)
    axs[j].spines['left'].set_visible(False)
    axs[j].set_xticks(ticks = np.linspace(0,52,27))
    axs[j].set_xticklabels(labels = (np.linspace(0,52,27,dtype=int)),fontsize=16, color = 'w')

custom_markers = [Line2D([0], [0], marker = "o", ms=22 , color=run, lw=0),
                Line2D([0], [0], marker = "o", ms=22 , color=ride, lw=0),
                Line2D([0], [0], marker = "o", ms=22 , color=hike, lw=0),
                Line2D([0], [0], marker = "o", ms=22 , color=alpineski, lw=0),
                Line2D([0], [0], marker = "o", ms=22 , color=nordicski, lw=0),
                Line2D([0], [0], marker = "o", ms=22 , color=canoe, lw=0)]
plt.legend(custom_markers, ['Run', 'Ride', 'Hike','Alpine Ski','Nordic Ski','Canoe'],
                      loc=(0.03,7.82),fontsize=20,labelcolor='w',facecolor='black')
plt.show()
```


![png](/assets/images/strava.png)



```python

```
