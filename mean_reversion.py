import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm


start = datetime.datetime(2007, 7, 22)
end = datetime.datetime(2012, 3, 28)

start = datetime.datetime(2007, 1, 1)
end = datetime.datetime(2024, 1, 1)


# EWA = yf.download("GC=F", start=start, end=end)
# EWC = yf.download("AUDNZD=X", start=start, end=end)
# IGE = yf.download("AUDUSD=X", start=start, end=end)

EWA = yf.download("EURUSD=X", start=start, end=end)
EWC = yf.download("AUDNZD=X", start=start, end=end)
IGE = yf.download("USDCHF=X", start=start, end=end)


w = IGE['Adj Close']
x = EWA['Adj Close']
y = EWC['Adj Close']
w.to_pickle('w')
x.to_pickle('x')
y.to_pickle('y')


df = pd.DataFrame([w,x,y]).transpose()
df.columns = ['W','X','Y']
df=df.dropna()
#df.plot(figsize=(20,12))


y3 = df

j_results = coint_johansen(y3,0,1)

print(j_results.lr1)
print(j_results.cvt)
print(j_results.eig)
print(j_results.evec)
print(j_results.evec[:,0])

hedge_ratios = j_results.evec[:, 0]
y_port = (hedge_ratios * df).sum(axis=1)
#y_port.plot(figsize=(20, 12))
y_port_lag = y_port.shift(1)
y_port_lag[0] = 0
delta_y = y_port - y_port_lag
X = y_port_lag
Y = delta_y
X = sm.add_constant(X)
model = sm.OLS(Y, X)
regression_results = model.fit()
regression_results.summary()

halflife = int(-np.log(2)/regression_results.params[0])
print(halflife)

num_units = -(y_port-y_port.rolling(halflife).mean())/y_port.rolling(halflife).std()
#num_units.plot(figsize=(20,12))


num_units = num_units.transpose()
df['Portfolio Units'] = num_units
df['W $ Units'] = df['W']*hedge_ratios[0]*df['Portfolio Units']
df['X $ Units'] = df['X']*hedge_ratios[1]*df['Portfolio Units']
df['Y $ Units'] = df['Y']*hedge_ratios[2]*df['Portfolio Units']
positions = df[['W $ Units','X $ Units','Y $ Units']]


positions = df[['W $ Units','X $ Units','Y $ Units']]
df5=df.iloc[:,0:3]
pnl=np.sum((positions.shift().values)*(df5.pct_change().values), axis=1)
ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
plt.figure(figsize=(8,5))
plt.plot(np.cumprod(1+ret)-1)
print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))