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

Strava is a activity tracking application that allows users to see their distance and route, and a whole host of other interesting activity data. Since I have been using this application for over 7 years now, I have built up a large cache of personal data in the app. To see how my activity in general and my activity of choice in particular have changed over the years, I created the following data visualizations.

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
df['Weekday'] = list(map(lambda x: x.weekday(), df['Activity Date'])) # weekdays labeled 0-6 meaning Monday-Sunday
daylist = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['weekday'] = list(map(lambda x: daylist[x], df['Weekday'])) # weekdays labeled Monday-Sunday
df['Distance mi'] = list(map(lambda x: x*0.621371, df['Distance']))
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


![png](/assets/images/strava1.png)


### Additional visualizations


```python
def group_and_count(colname):
    count_df = df.groupby([colname]).count()['Activity ID']
    count_df = count_df.sort_values(ascending=False)
    x_labs = np.array(count_df.index)
    y_vals = np.array(count_df)
    return x_labs,y_vals

def charter(xlabs,y,title):
    plt.figure(figsize=(10,6))
    xpoints = np.linspace(start = 0, stop = len(y)-1,num = len(y))
    plt.bar(x = xpoints, height = y, width = 0.9, tick_label = xlabs, color = 'teal')
    plt.title(title)
    plt.xticks(fontsize=16)
    plt.grid(None,axis='x')
    for count, yval in enumerate(y):
        plt.annotate(text = f"{yval}", xy = (count-0.15,yval+5),fontsize=14)
    plt.show()
```


```python
results = group_and_count('Activity Type')
charter(results[0],results[1],'Activity Count by Type (2014-2021)')
```


![png](/assets/images/strava2.png)



```python
results = group_and_count('Year')
charter(results[0],results[1],'Activity Count by Year (2014-2021)')
```


![png](/assets/images/strava3.png)



```python
results = group_and_count('weekday')
charter(results[0],results[1],'Activity Count by Weekday (2014-2021)')
```


![png](/assets/images/strava4.png)



```python
avg_dist = df.groupby(['Activity Type']).mean()['Distance mi'].sort_values(ascending=False)

activity_count_dict = df.groupby(['Activity Type']).count()['Activity ID'].to_dict() # dict for annotations

y = avg_dist
xlabs = avg_dist.index
plt.figure(figsize=(10,6))
xpoints = np.linspace(start = 0, stop = len(y)-1,num = len(y))
plt.bar(x = xpoints, height = y, width = 0.9, tick_label = xlabs, color = 'teal')
plt.title("Average Distance by Activity Type (2014-2021)")
plt.ylabel("Distance (mi)")
plt.xticks(fontsize=16)
plt.grid(None,axis='x')
for count, yval in enumerate(y):
    plt.annotate(text = f"{yval:.2f}", xy = (count-0.15,yval+0.5),fontsize=14)
    plt.annotate(text = f"n = {activity_count_dict[xlabs[count]]}", xy = (count-0.3,yval-1),fontsize=14)
plt.show()
```


![png](/assets/images/strava5.png)



```python
def specific_df(activity):
    specific_df = df.where(df['Activity Type']==activity,np.nan)
    return specific_df.dropna(axis=0,how='all').reset_index()
```


```python
ride_df = specific_df('Ride')
run_df = specific_df('Run')
ski_df = specific_df('Nordic Ski')
hike_df = specific_df('Hike')
```


```python
plt.figure(figsize=(10,6))
plt.boxplot([ride_df['Distance mi'],run_df['Distance mi'],ski_df['Distance mi'],hike_df['Distance mi']],
            labels = ['Ride','Run','Nordic Ski','Hike'])
plt.ylabel("Distance (mi)")
plt.title("Average Distance by Activity Type (2014-2021)")
plt.show()
```


![png](/assets/images/strava6.png)



```python
plt.figure(figsize=(10,8))
plt.hist([ride_df['Distance mi'],run_df['Distance mi'],ski_df['Distance mi'],hike_df['Distance mi']],
         histtype = 'step', alpha = 1.0, range = (0,60), bins=60,
         label = ["Ride","Run",'Nordic Ski','Hike'],color = ['blue','red','teal','orange'],lw=1.0)
plt.xlim(-1,30)
plt.title("Distance Distribution by Activity Type (2014-2021)")
plt.xlabel("Distance (mi)")
plt.ylabel("Occurrences")
plt.legend()
plt.show()
```


![png](/assets/images/strava7.png)



```python
plt.figure(figsize=(10,8))
plt.hist([ride_df['Distance mi'],run_df['Distance mi'],ski_df['Distance mi'],hike_df['Distance mi']],
         histtype = 'step', alpha = 1.0, range = (0,60), bins=120, cumulative = True, density = True,
         label = ["Ride","Run",'Nordic Ski','Hike'],color = ['blue','red','teal','orange'],lw=1.0)
plt.xlim(-1,20)
plt.title("Empirical Cumulative Distribution Function\nby Activity Type (2014-2021)")
plt.xlabel("Distance (mi)")
plt.ylabel("Probability")
plt.legend(loc='lower right')
plt.show()
```


![png](/assets/images/strava8.png)


Just for interest, I'll create a simple model to calculate the probability that any of my rides is greater than a certain distance.


```python
from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(ride_df['Distance mi']) # create model with ride data
```


```python
print(f"*Based on my personal riding history*")
print(f"Distance: 5 miles  \t Probability of Riding Farther: {1-ecdf(5):.4f}")
print(f"Distance: 10 miles \t Probability of Riding Farther: {1-ecdf(10):.4f}")
print(f"Distance: 15 miles \t Probability of Riding Farther: {1-ecdf(15):.4f}")
print(f"Distance: 20 miles \t Probability of Riding Farther: {1-ecdf(20):.4f}")
print(f"Distance: 30 miles \t Probability of Riding Farther: {1-ecdf(30):.4f}")
```

    *Based on my personal riding history*
    Distance: 5 miles  	 Probability of Riding Farther: 0.8468
    Distance: 10 miles 	 Probability of Riding Farther: 0.4393
    Distance: 15 miles 	 Probability of Riding Farther: 0.1908
    Distance: 20 miles 	 Probability of Riding Farther: 0.0838
    Distance: 30 miles 	 Probability of Riding Farther: 0.0260

