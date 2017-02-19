#%%
# Read in data
import pandas as pd
import numpy as np
# Download CSV from https://goo.gl/zJN9rm
df = pd.read_csv(r'C:/Users/alex/Downloads/interest.csv')

#%%
# Compute z-scores
# thanks http://stackoverflow.com/a/24762240
cols = set(['PA', 'GB%', 'LD%', 'G', 'OPS', 
'IFFB%', 'BB%', 'AVG', 'wRC+', 'BABIP', 'K%', 
'wOBA', 'ISO', 'FB%', 'SLG', 'WAR', 'OBP'])

# Z-scores and squared z-scores
zscore2_cols = []
for col in cols:
    col_zscore = col + '_zscore'
    colz2 = col + '_zscore2'
    zscore2_cols.append(colz2)
    df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    df[colz2] = ((df[col] - df[col].mean())/df[col].std(ddof=0))**2
 
#%%
# Sum the _zscore2 columns and find the min
mysum = df.loc[:,zscore2_cols].sum(axis=1)

# Who was the least interesting?
ixmin = np.argmin(mysum)
print df.loc[ixmin,['Name']]

# Who was the most interesting?
ixmax = np.argmax(mysum)
print df.loc[ixmax,['Name']]

#%%
# Make the CSV for the histogram
df['zscore_square_sum'] = mysum
df2 = df.loc[:,['Name','zscore_square_sum']]
df2 = df2.sort(['zscore_square_sum'])
df2.to_csv('histogram.csv',index=False)

#%%
# Try doing some scatter plots
import matplotlib.pyplot as plt

stats = ('FB%','wOBA')
plt.plot(df.loc[:,[stats[0]]],df.loc[:,[stats[1]]],'o')
plt.plot(df.loc[ixmin,[stats[0]]],df.loc[ixmin,[stats[1]]],'ro')
plt.xlabel()
plt.show()

#%%
# Where was Piscotty the most interesting?
df3 = df.loc[ixmin,zscore2_cols]
#df3.to_csv('Piscotty_scores.csv',index=True)
