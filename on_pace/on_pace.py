#%%
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
import numpy as np
#%%
# Read in the CSV
# From https://github.com/chadwickbureau/baseballdatabank/blob/master/core/Teams.csv
r = pd.read_csv(r'Teams.csv')
# Restrict to just post 1961 (162-game schedule)
r = r.loc[r.yearID>=1961]
# Add a column for win percentage
r['WinPct']=r['W']/(r['W']+r['L'])
# Make histogram
ydata,xdata,_ = plt.hist(r['WinPct'],bins=75)
plt.title('Historical Winning Percentages')
plt.xlabel('Win Percentage')
plt.ylabel('Number of Teams')
plt.savefig('winpct.png',dpi=300)

#%%
# Mean and variance of win percentages
mu = r['WinPct'].mean()
v = r['WinPct'].var()

# Estimate alpha and beta from these
# Thanks https://stats.stackexchange.com/a/12239
alpha = mu**2 * ((1-mu)/v - 1/mu)
b = alpha * (1/mu - 1)

# Plot the beta distribution along with the normalized histogram
plt.hist(r['WinPct'],bins=75,normed=True)
x = np.linspace(0.2,0.8,num=100)
y = beta.pdf(x,alpha,b)
plt.plot(x,y,color='red',linewidth=3)
plt.savefig('beta.png',dpi=300)

#%%
# Update the prior and plot
alpha2 = alpha + 10
beta2 = b
y2 = beta.pdf(x,alpha2,beta2)
plt.plot(x,y,color='red',label='Prior')
plt.plot(x,y2,color='green',label='Posterior')
plt.legend(loc='upper left')
plt.savefig('posterior.png',dpi=300)

# New mean?
mean2 = alpha2/(alpha2+beta2)
print mean2
print mean2*162

# New mode
mymode = (alpha2-1)/(alpha2+beta2-2)
print mymode*162