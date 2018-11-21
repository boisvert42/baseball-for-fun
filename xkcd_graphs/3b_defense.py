import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import spline
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
 
#%%
# TZ numbers via https://www.fangraphs.com/leaders.aspx?pos=3b&stats=fld&lg=all&qual=y&type=0&season=2017&month=0&season1=1961&ind=1&team=0&rost=0&age=0&filter=&players=0
df = pd.read_csv(r'TZ.csv')
df = df[['Season','Name','TZ','playerid']]
# For Total Zone delete anything past 2002
df = df.loc[df.Season < 2002]
 
# UZR numbers via https://www.fangraphs.com/leaders.aspx?pos=3b&stats=fld&lg=all&qual=y&type=1&season=2017&month=0&season1=2002&ind=1&team=0&rost=0&age=0&filter=&players=0
df2 = pd.read_csv(r'UZR.csv')
df2 = df2[['Season','Name','UZR','playerid']]
df2.columns = ['Season','Name','TZ','playerid']
 
df = pd.concat([df,df2])
df = df.sort_values(by='Season')
 
#%%
# Figure out which names to use
# by looking at max TZ for each year
gb = df.groupby(['Season'])['TZ'].max()
gb = gb.reset_index()
gb = gb.sort_values(by='Season')
df3 = df.merge(gb)
df3 = df3.sort_values(by='Season')
df3 = df3.loc[df3['TZ'] > 15]
 
names = ['Brooks Robinson', 'Graig Nettles', 'Mike Schmidt', 'Buddy Bell',
       'Gary Gaetti', 'Terry Pendleton', 'Robin Ventura',
       'Scott Rolen', 'Adrian Beltre', 'Evan Longoria', 'Manny Machado']

colors = ['blue','lightblue','lightblue','blue','lightblue','lightblue','blue','lightblue','gold','lightblue','lightblue']

 
fig = plt.figure()

SPLINE_FACTOR = 1.6
 
# Go through and plot each player
for i,name in enumerate(names):
    #print(name)
    col = colors[i]
    linewidth = 1
    if col != 'lightblue':
        linewidth=2
   
    df_name = df.loc[df['Name'] == name]
    df_name.index = range(len(df_name))
 
    xnew = np.linspace(df_name.Season.min(),df_name.Season.max(),300) #300 represents number of points to make between T.min and T.max
 
    #tz_smooth = spline(df_name.Season,df_name.TZ,xnew)
    tz_smooth = spline(df_name.Season,gaussian_filter1d(df_name.TZ,SPLINE_FACTOR),xnew)
    tz_convolved = np.convolve(df_name.TZ,np.ones(3,)/3,mode='same')
    tz_smooth2 = spline(df_name.Season,gaussian_filter1d(tz_convolved,SPLINE_FACTOR),xnew)
   
    #plt.plot(xnew,tz_smooth,label=name)#,color=col)
    #plt.plot(df_name['Season'],df_name['TZ'],label=name)#,color=col)
    #plt.plot(df_name['Season'],tz_convolved,label=name)#,color=col)
    plt.plot(xnew,tz_smooth2,label=name,color=col,linewidth=linewidth)
    
    #print df_name['TZ']
# Axis labeling and such
#plt.legend(fontsize=8,loc='lower left')
plt.ylabel('TZ/UZR')
plt.xlabel('Year')
plt.xlim((1960,2017))
plt.title('Third Base Defense')
plt.tight_layout()
   
# Save the plot as PNG
filename = '3b_defense.png'
fig.savefig(filename,dpi=400)
