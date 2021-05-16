#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprint Speed Prediction

Can we modify Bill James's Spd calculation to be in line with the Statcast numbers?
"""

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

#%% Gather the data
# Fangraphs data
# https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=40&type=c,21,22,8,14,15,17,5,6,11,16,7,12,10,20,60&season=2019&month=0&season1=2017&ind=1&team=0&rost=0&age=0&filter=&players=0&startdate=&enddate=

df_fangraphs = pd.read_csv('fangraphs.csv')

# To avoid confusion (and because the Statcast player ID != Fangraphs player ID)
# we just drop any players with overlapping names
df_fangraphs.drop_duplicates(subset=['Season', 'Name'], keep=False, inplace=True)

# Statcast data for 2017-2019
# https://baseballsavant.mlb.com/leaderboard/sprint_speed?year=2019&position=&team=&min=10
arr_statcast = []
for yr in (2017, 2018, 2019):
    df_tmp = pd.read_csv(f'sprint_speed_{yr}.csv', skipinitialspace=True)
    # add a "Name" and "Season" column
    df_tmp['Name'] = df_tmp['first_name'] + ' ' + df_tmp['last_name']
    df_tmp['Season'] = yr
    arr_statcast.append(df_tmp)
df_statcast = pd.concat(arr_statcast)

df = pd.merge(df_fangraphs, df_statcast, on=['Name', 'Season'])
df['Season'] = pd.Categorical(df['Season'])

#%% Initial analysis
# How well does the original speed score match up with sprint speed?
fig = px.scatter(df, x="Spd", y="sprint_speed", 
                 color="Season", hover_data=["Name", "Season"],
                 title="Sprint Speed vs. Speed Score")
fig.show()
plotly.offline.plot(fig, filename='initial_scatter.html')

# How's the R-squared of the fit?
print(np.corrcoef(df['Spd'], df['sprint_speed'])[0,1]**2)
# 0.42 is our baseline

###
# Now let's go through the individual components to see how we can improve
# https://en.wikipedia.org/wiki/Speed_Score#Factors
###

REGRESSION_COLUMNS = []

#%% Stolen base percentage
# The original calculation used (3,7) to regress
# Let's try a few different values
regressors = []
dfc = df.copy()
for denom in range(2, 16):
    for num in range(1, denom):
        regressors.append((num, denom))
        dfc[f'sbp_{num}_{denom}'] = (dfc['SB'] + num)/(dfc['SB'] + dfc['CS'] + denom)

sbp_corr = dfc.corr()['sprint_speed']
# I'm surprised but the correlation analysis suggests we should
# regress VERY heavily
# we'll use 1 / 10
df['sb_perc_regressed'] = (dfc['SB'] + 1)/(dfc['SB'] + dfc['CS'] + 10)
REGRESSION_COLUMNS.append('sb_perc_regressed')

#%% Stolen base attempts
# James takes the square root.
# Is that better than not?
dfc = df.copy()
for exp1 in range(1, 20):
    exp = exp1/10
    dfc[f'sb_{exp1}'] = ((df['SB'] + df['CS'])/(df['1B'] + df['BB'] + df['IBB'] + df['HBP']))**exp

sb_corr = dfc.corr()['sprint_speed']
# 0.6 is slightly better but let's just go with Bill's calculation
df['sb_sqrt'] = ((df['SB'] + df['CS'])/(df['1B'] + df['BB'] + df['IBB'] + df['HBP']))**0.5
REGRESSION_COLUMNS.append('sb_sqrt')

#%% Triples
# Is there any benefit in regressing this number?
dfc = df.copy()
for denom in range(2, 50):
    for num in range(1, 2):
        dfc[f'3b_{num}_{denom}'] = (df['3B'] + num)/(df['AB'] - df['HR'] - df['SO'] + denom)

# It looks like there is a benefit to heavily regressing this number
# but we'll just go with the original
corr_3b = dfc.corr()['sprint_speed']
df['triples'] = df['3B']/(df['AB'] - df['HR'] - df['SO'])
REGRESSION_COLUMNS.append('triples')

#%% Runs scored
# Let's just add James's number
df['runs_scored'] = (df['R'] - df['HR'])/(df['H'] + df['BB'] + df['IBB'] + df['HBP'] - df['HR'])
REGRESSION_COLUMNS.append('runs_scored')

#%% GDP
# I'm not sure why James does 0.063 minus this number
df['double_plays'] = df['GDP']/(df['AB'] - df['HR'] - df['SO'])
REGRESSION_COLUMNS.append('double_plays')

# Note: we don't have positional information from Fangraphs :(

#%% Linear regression
y = df['sprint_speed']
X = df[REGRESSION_COLUMNS]
model = LinearRegression()
model.fit(X, y)
r_sq = model.score(X, y)
print(r_sq)
# 0.468 r-squared: not that much better
y_pred = model.predict(X)

df_lin = pd.DataFrame(data=y, columns=['sprint_speed'])
df_lin['sprint_speed_predicted'] = y_pred

fig2 = px.scatter(df_lin, x="sprint_speed_predicted", y="sprint_speed", 
                 title="Sprint Speed vs. Predicted Value")

plotly.offline.plot(fig2, filename='linear_prediction.html')

#%% Does a more sophisticated algorithm help?

df_all = df[REGRESSION_COLUMNS + ['sprint_speed', 'Name', 'Season']]

train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=10)

X_train = np.array(train_df[REGRESSION_COLUMNS])
y_train = np.array(train_df['sprint_speed'])

X_test = np.array(test_df[REGRESSION_COLUMNS])
y_test = np.array(test_df['sprint_speed'])

regr = RandomForestRegressor(max_depth=5, random_state=0)
regr.fit(X_train, y_train)

test_predict = regr.predict(X_test)
plt.plot(test_predict, y_test, '.')

print(np.corrcoef(test_predict, y_test)[0,1]**2)
# 0.49 r-squared -- only slightly better
