import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from matplotlib import cm

#%%
# TZ numbers via https://www.fangraphs.com/leaders.aspx?pos=cf&stats=fld&lg=all&qual=y&type=0&season=2019&month=0&season1=1956&ind=1&team=0&rost=0&age=0&filter=&players=0&startdate=&enddate=
df = pd.read_csv(r'TZ.csv')
df = df[['Season','Name','TZ','playerid']]
# For Total Zone delete anything past 2002
df = df.loc[df.Season < 2002]
 
# UZR numbers via https://www.fangraphs.com/leaders.aspx?pos=cf&stats=fld&lg=all&qual=y&type=1&season=2019&month=0&season1=2002&ind=1&team=0&rost=0&age=0&filter=&players=0&startdate=&enddate=
df2 = pd.read_csv(r'UZR.csv')
df2 = df2[['Season','Name','UZR','playerid']]
df2.columns = ['Season','Name','TZ','playerid']
 
df = pd.concat([df,df2])
df = df.sort_values(by='Season')

#%%
# One complication with the above is that we don't have early Willie Mays numbers
# Let's try to correlate his putout numbers with his TZ numbers
# Putout numbers for pre-2016 from https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/Fielding.csv
# This takes a long time so we don't do it unless we need to
if False:
    df_po = pd.read_csv(r'https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/Fielding.csv')
    df_wm = df_po.loc[(df_po['playerID'] == 'mayswi01') & (df_po['POS'] == 'OF')]
    df_wm_tz = df.loc[df['Name'] == 'Willie Mays']
    df_wm_tz.columns = ['yearID', 'Name', 'TZ', 'playerid']
    df_wm_join = pd.merge(df_wm, df_wm_tz, on='yearID')
    
    #print(df_wm_join[['TZ', 'PO']])
    
    plt.plot(df_wm_join['TZ'], df_wm_join['PO'], 'o')
    
    # Not a perfect relationship but a linear fit should be fine
    model = LinearRegression()
    x = [[_] for _ in df_wm_join['PO']]
    y = df_wm_join['TZ']
    model.fit(x, y)
    r_sq = model.score(x, y)
    # Make predictions for his earlier data
    y1 = model.predict([[_] for _ in df_wm.head(4)['PO']])
# No data point for 1953, and the one for 1952 is too low, so we'll fudge it
# Let's set his values to [9, 12, 15, 17, 14]
#%%
new_wf_tz = [9, 12, 15, 17, 14]
new_data = []
for i, tz in enumerate(new_wf_tz):
    yr = i + 1951
    new_data.append([yr, 'Willie Mays', tz, 1008315])
    
df_wm_new = pd.DataFrame(data=new_data, columns=df.columns)
df = pd.concat([df, df_wm_new])
df = df.sort_values(by='Season')

#%%
# Figure out which names to use
# by looking at max TZ for each year
gb = df.groupby(['Season'])['TZ'].max()
gb = gb.reset_index()
gb = gb.sort_values(by='Season')
df3 = df.merge(gb)
df3 = df3.sort_values(by='Season')
df3 = df3.loc[df3['TZ'] > 10]
 
names = ['Willie Mays', 'Jim Piersall', 'Paul Blair',
         'Garry Maddox', 'Andre Dawson',
         'Kirby Puckett', 'Gary Pettis',
         'Devon White', 'Kenny Lofton', 'Andruw Jones',
         'Mike Cameron',  'Michael Bourn']

colors = ['blue', 'lightblue', 'lightblue',
           'blue', 'lightblue',
          'lightblue', 'blue',
          'blue', 'lightblue', 'gold', 
          'blue', 'lightblue']

#colors = cm.get_cmap('tab20').colors

fig = plt.figure()

GAUSS_FACTOR = 2
 
# Go through and plot each player
for i,name in enumerate(names):
    #print(name)
    col = colors[i]
    linewidth = 1
    if col != 'lightblue':
        linewidth=2
   
    df_name = df.loc[df['Name'] == name]
    df_name.index = range(len(df_name))
    
    x = df_name.Season
    y = gaussian_filter1d(df_name.TZ,GAUSS_FACTOR)
    f = interp1d(x, y, kind='cubic')
 
    xnew = np.linspace(df_name.Season.min(),df_name.Season.max(),300) #300 represents number of points to make between T.min and T.max
   
    plt.plot(xnew,f(xnew),label=name,color=col,linewidth=linewidth)
    
    #print df_name['TZ']
# Axis labeling and such
#plt.legend(fontsize=8,loc='lower left')
plt.ylabel('UZR')
plt.xlabel('Year')
plt.xlim((1951,2015))
#plt.ylim([0,32])
plt.title('CF Defense')
plt.tight_layout()
plt.legend()
   
# Save the plot as PNG
filename = 'cf_defense.png'
fig.savefig(filename,dpi=400)

