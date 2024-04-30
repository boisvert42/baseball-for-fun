# Data exported from Fangraphs

import json
import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import cm

# Load the data
df = pd.read_csv(r'fangraphs-leaderboards.csv')

#%% Look at and pre-process data
df = df.sort_values(by='WAR', ascending=False)
df1 = df.head(10)

df = df.sort_values(by=['Name', 'Season'])
df = df.loc[df['Season'] < 2024]


#%% Choose players
allYears = df['Season'].unique()
for yr in sorted(allYears):
    df_tmp = df.loc[df['Season'] == yr].copy()
    df_tmp = df_tmp.sort_values(by='WAR', ascending=False)
    df_tmp.reset_index(inplace=True)
    df_tmp = df_tmp.head(2)
    print(yr, df_tmp['Name'].unique())

#%% helper function for window averages
def window_avg(arr, window_size=3):
    N = len(arr)
    M = (window_size - 1)//2
    arr1 = [arr[0]] * M + arr + [arr[-1]] * M
    ret = []
    for i in range(N):
        arrTmp = arr1[i:i+window_size]
        ret.append(sum(arrTmp)/len(arrTmp))
    return ret
    

#%%
# Restrict to certain names
names = ['Toad Ramsey', 
         'Ted Breitenstein',
         #'Noodles Hahn',
         'Rube Waddell',
         #'Eddie Plank',
         #'Rube Marquard', 
         'Hippo Vaughn',
         'Wilbur Cooper',
         'Lefty Grove', 
         #'Carl Hubbell',
         'Hal Newhouser',
         'Warren Spahn',
         'Whitey Ford',
         'Sandy Koufax',
         'Sam McDowell',
         'Steve Carlton',
         #'Fernando Valenzuela',
         'Frank Viola',
         'Tom Glavine',
         'Randy Johnson',
         'Johan Santana',
         #'Andy Pettitte',
         'Clayton Kershaw'
         ]

colors = cm.get_cmap('tab20').colors
markers = [x[0] for x in Line2D.markers.items()]

fig = plt.figure()

SPLINE_FACTOR = 1.3

plt.rc('axes', labelsize=15) #fontsize of the x and y labels
plt.rc('legend', fontsize=10) #fontsize of the legend
plt.rc('axes', titlesize=15) #fontsize of the x and y labels
 
# Go through and plot each player
for i,name in enumerate(names):
    print(name)
    col = colors[i]
    linewidth = 1
    if name in ('Toad Ramsey', 'Sandy Koufax', 'Randy Johnson', 
                'Clayton Kershaw', 'Hal Newhouser',
                'Steve Carlton', 'Rube Waddell', 'Lefty Grove'):
        linewidth=2
        col = 'slateblue'
    else:
        col = 'gray'
   
    df_name = df.loc[df['Name'] == name]
    df_name.index = range(len(df_name))
    
    x = df_name['Season'].values.tolist()
    y = df_name['WAR'].values.tolist()

    xnew = np.linspace(min(x), max(x), 300)
    ynew = gaussian_filter1d(y, SPLINE_FACTOR)
    #ynew = window_avg(y)
    #ynew = y
    f = interp1d(x, ynew, kind='cubic')
    ynew2 = f(xnew)
    
    #Smoothed with markers
    #plt.plot(x, ynew, label=name, color=col, marker=markers[i], linewidth=linewidth)
    # Unsmoothed
    #plt.plot(x, y, label=name, color=col, marker=markers[i], linewidth=linewidth)
    #Smoothed, no markers
    plt.plot(xnew, ynew2, label=name, color=col, linewidth=linewidth)
    
# Axis labeling and such
#plt.legend(fontsize=8,loc='lower left')
plt.ylabel('fWAR')
plt.xlabel('Year')
plt.xlim((1885, 2022))
plt.title('Dominant Left-Handed Pitchers')
plt.tight_layout()
#plt.legend(loc='lower left')

   
# Save the plot as PNG
filename = 'lhp.png'
fig.savefig(filename,dpi=400)

