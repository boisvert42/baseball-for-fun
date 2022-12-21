"""
https://api.nhle.com/stats/rest/en/goalie/advanced?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22savePct%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=100&factCayenneExp=gamesPlayed%3E=1%20and%20gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20212022%20and%20seasonId%3E=19171918
"""

import json
import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import cm

url = '''https://api.nhle.com/stats/rest/en/goalie/advanced?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22savePct%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start={0}&limit=100&factCayenneExp=gamesPlayed%3E=1%20and%20gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20212022%20and%20seasonId%3E=19171918'''

data = 1
rawData = []
start = 0

if False:
    # try to get data from NHL API
    while data:
        print(start)
        url1 = url.format(start)
        data1 = requests.get(url1).json()
        data = data1['data']
        rawData = rawData + data
        start += 100
        time.sleep(2)
        
    with open(r'goalie.json', 'w') as fid:
        json.dump(rawData, fid)
else:
    with open(r'goalie.json', 'r') as fid:
        rawData = json.load(fid)

#%%
df = pd.DataFrame(data=rawData)

# compute some helper statistics
df['shots'] = df['goalsAgainst']/(1-df['savePct'])
df['saves'] = df['shots'] - df['goalsAgainst']
df['year'] = df['seasonId'].apply(lambda x:int(str(x)[:4]))

df = df.sort_values(by='savePct')
df1 = df.head(10)

# Compute league-average save percentage per year
allYears = df['year'].unique()
avgSavePctByYear = dict()
for yr in sorted(allYears):
    df_tmp = df.loc[df['year'] == yr].copy()
    avgSavePct = df_tmp['saves'].sum()/df_tmp['shots'].sum()
    if not pd.isna(avgSavePct):
        avgSavePctByYear[yr] = avgSavePct

#%% Make an initial plot of this data per year
plt.figure()
yrs = sorted(avgSavePctByYear.keys())
savePcts = [avgSavePctByYear[yr] for yr in yrs]
plt.plot(yrs, savePcts)
plt.xlabel('Year')
plt.ylabel('Save Percentage')
plt.title('Save Percentages by Year')

#%% Compute GSAA
df['GSAA'] = df['saves'] - df['year'].apply(lambda x: avgSavePctByYear.get(x, 0)) * df['shots']
df = df.sort_values(by='GSAA', ascending=False)
df1 = df.head(100)

print(df1['goalieFullName'].unique())

# Restrict to years with actual values
minYear = df.loc[df['GSAA'] > 0]['year'].min()
dfFinal = df.loc[df['year'] >= minYear].copy()

dfFinal.sort_values(by='year', inplace=True)

#%% Choose players
allYears = dfFinal['year'].unique()
for yr in sorted(allYears):
    df_tmp = dfFinal.loc[dfFinal['year'] == yr].copy()
    df_tmp = df_tmp.sort_values(by='GSAA', ascending=False)
    df_tmp.reset_index(inplace=True)
    df_tmp = df_tmp.head(2)
    print(yr, df_tmp['goalieFullName'].unique())

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
names = [#'John Vanbiesbrouck', 
         'Curtis Joseph', 'Dominik Hasek',
         #'Rogie Vachon', 
         'Jacques Plante',
         'Tony Esposito', 'Ken Dryden',
         'Roberto Luongo', 
         #'Jose Theodore', 
         #'Marty Turco',
         'Martin Brodeur', #'Felix Potvin',
         #'Pekka Rinne', 
         'Grant Fuhr', 'Patrick Roy',# 'Tom Barrasso',
         'Glenn Hall', 
         #'Tim Thomas', 
         'Carey Price', #'Gump Worsley',
         'Johnny Bower', 
         #'Ed Belfour', 
         'Billy Smith',
         #'Terry Sawchuk', 'Corey Crawford',
         'Henrik Lundqvist', 
         'Jonathan Quick', 'Marc-andre Fleury',
         'Jean-Sebastien Giguere', 
         'Rick DiPietro',
         'Bernie Parent', 'Andrei Vasilevskiy', 'Andrew Hammond']

#names = ['Carey Price', 'Jonathan Quick']


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
    if name in ('Ken Dryden', 'Tony Esposito', 'Patrick Roy', 
                'Dominik Hasek', 'Billy Smith', 'Andrew Hammond', 'Johnny Bower', 
                'Curtis Joseph', 'Marc-andre Fleury',
                'Roberto Luongo', 'Andrei Vasilevskiy'):
        linewidth=2
        #col = 'slateblue'
    #else:
    #    col = 'gray'
   
    df_name = dfFinal.loc[df['goalieFullName'] == name]
    df_name.index = range(len(df_name))
    
    x = df_name['year'].values.tolist()
    y = df_name['GSAA'].values.tolist()
    if name == 'Patrick Roy':
        x = x[1:]; y = y[1:]

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
plt.ylabel('GSAA')
plt.xlabel('Year')
plt.xlim((minYear, 2021))
plt.title('Dominant Goalies')
plt.tight_layout()
plt.legend(loc='upper left')

   
# Save the plot as PNG
#filename = 'goalie.png'
#fig.savefig(filename,dpi=400)

