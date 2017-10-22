import numpy as n
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests, ccf, adfuller, acf
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
fullsize = 46
windowsize = 5

# define an autocorrelation function, which compares k number of time series
def autocorr(x):
    nn = len(x)
    variance = x.var()
    x = x-x.mean()
    r = n.correlate(x, x, "full")[-nn:]
    assert n.allclose(r, n.array([(x[:nn-k]*x[-(nn-k):]).sum() for k in range(nn)]))
    result = r/(variance*(n.arange(nn, 0, -1)))
    return result

# read from csv file and set up data for computations
dataFile = pd.read_csv('../Data/RegressionData.csv')
X, Y = list(dataFile['Bullishness_N'].iloc[1:].values.flatten()),list(dataFile['Log_Return'].iloc[1:].values.flatten())
data2d = n.array([Y,X])
print data2d


# Compute VAR, Augmented Dickey-Fuller
sizedata = n.transpose(data2d)
spread = [el[1] - el[0] for el in sizedata]
vartest = VAR(sizedata)
gtest = grangercausalitytests(sizedata,maxlag=14, addconst=True, verbose=True)
adftest = adfuller(spread,regression='ct',regresults=True)
adfX, adfY = adfuller(X,regression='ct',regresults=True), adfuller(Y,regression='ct', regresults=True)
varresult = vartest.fit(10)


# Output Results and create charts
print varresult.summary()
print vartest.select_order(10)
print varresult.test_causality(vartest, sizedata,kind='f')
print acf(spread, qstat=True)
plot_acf(spread)
plt.title('Spread Autocorrelation')
plt.savefig('5.png')
print adftest

#show spreads
for s in gtest:
    print s
for key, value in adftest[2].items():
	print('Critical Values at %s: %.3f' % (key, value))
print '\nADF Statistic on (X-Y): %f' % adftest[0]
print('p-value: %f' % adftest[1])

#show X
print '\nADF Statistic on Bullishness Indicator Series: %f' % adfX[0]
print('p-value: %f' % adfX[1])
	
#show Y
print '\nADF Statistic on ITD Return of Bitcoin: %f' % adfY[0]
print('p-value: %f' % adfY[1])

#create OLS
index = [i for i in range(1,len(X))]
ols_X = sm.add_constant(X)
olsModel = sm.OLS(Y,ols_X)
olsResult = olsModel.fit()
olsPredict = olsResult.predict()
olsResidual = olsResult.fittedvalues
adfResidual = adfuller(Y-n.transpose(olsResidual),regression='ct', regresults=True)
print'\nADF Statistic on Residuals of X and Y: %f' % adfResidual[0]
print('p-value: %f' % adfResidual[1])
df = pd.DataFrame()
df['BI'] = X
df['BTC ITD'] = Y
df['Predict'] = olsPredict
df['Res'] =Y- olsResidual
print df
ar= autocorr(Y-olsResidual)
nindex = [i for i in range(1,len(X)+1)]

#creating charts now
plt.figure()
plt.title('Autocorrelation for OLS residuals')
plt.bar(nindex,ar)
plt.legend(['Autocorrelation(k)'])
plt.xlabel('Number of Days')
plt.savefig('4.png')
plt.clf()
durbinw = durbin_watson(Y-olsResidual)
print durbinw
spread_mean = [n.mean(spread) for i in spread]
plt.figure()
plt.title('Bullishness Indicator Vs ITD Return of Bitcoin Prices')
plt.plot(spread,'b')
plt.plot(spread_mean, 'r')
plt.legend(['Spread','Spread_Mean'])
plt.xlabel('Number of Days')
plt.savefig('3.png')
plt.clf()

# Plot Covariance
X = X - sum(X)/len(X)
Y = Y - sum(Y)/len(Y)
Z = Y.copy();
Z[1:] = Z[1:] + X[0:-1]

# so: X influences Z with timelag 1
covarXY = n.zeros(fullsize-windowsize)
for i in range(0,fullsize-windowsize):
    EX = X[i:i+windowsize]
    EX2 = sum(EX*EX)/(windowsize+1)
    EX = sum(EX)/(windowsize+1)
    EZ = Z[i:i+windowsize]
    EZ2 = sum(EZ*EZ)/(windowsize+1)
    EZ = sum(EZ)/(windowsize+1)
    covarXY[i] = (sum(X[i:i+windowsize]*Z[i:i+windowsize])/(windowsize+1) - EX*EZ)/(math.sqrt((EX2-EX*EX)*(EZ2-EZ*EZ)))

plt.figure()
plt.title('Covariance Plot Between Bullish_Norm & ITD_Log_Return')
plt.plot(X, 'g', label = 'Bullish_Norm')
plt.plot(Y, 'r', label = 'ITD_Return')
plt.plot(covarXY, 'b', label = "Covariance")
plt.legend(['Bullish_Norm','ITD_Return','Covariance'])
maxval = max(max(X[0:100]),max(Z[0:100]))
minval = min(min(X[0:100]),min(Z[0:100]))

plt.xlabel('Number of Days')
plt.ylabel('Indicator & Return Data')
plt.savefig('1.png')
plt.clf()
A = sum(Z**2)/fullsize
B = sum(Z[0:-1]*Z[1:])/(fullsize-1);
C = sum(Z[0:-2]*Z[2:])/(fullsize-2);

a2 = (2*A*C-B)/(4*A**2-B**2);
a1 = (1-a2)*B/(2*A);

windowpredictionerror = n.zeros(fullsize-2)
windowpredictionerrorcause = covarXY = n.zeros(fullsize-windowsize-2)
windowpredictionerror = Z[2:] - a1*Z[1:-1] - a2*Z[:-2]
windowpredictionerrorcause = windowpredictionerror - X[1:-1]
wpeplot = n.zeros(fullsize-2-windowsize+1)
wpecplot = n.zeros(fullsize-2-windowsize+1)

for i in range(0,fullsize-2-windowsize):
    wpemean = sum((windowpredictionerror[i:i+windowsize]))/(windowsize+1)
    wpeplot[i] = sum((windowpredictionerror[i:i+windowsize] - wpemean)**2)/(windowsize+1)
    wpemean = sum((windowpredictionerrorcause[i:i+windowsize]))/(windowsize+1)
    wpecplot[i] = sum((windowpredictionerrorcause[i:i+windowsize] - wpemean)**2)/(windowsize+1)

plt.figure()
plt.title('Predictions for Bullish_Norm & ITD_Log_Return')
plt.plot(wpeplot[0:100], 'r', label="Prediction without X")
plt.plot(wpecplot[0:100], 'g', label="Prediction wih X")
maxval = max(max(wpeplot[0:100]), max(wpecplot[0:100]))
minval = min(min(wpeplot[0:100]), min(wpecplot[0:100]))
plt.legend(['Prediction without X','Prediction with X'])
plt.xlabel('Number of Days')
plt.ylabel('Scores')
plt.savefig('2.png')
predictionerror = n.zeros(fullsize-2)
predictionerror = Z[2:] - a1*Z[1:-1] - a2*Z[:-2]
variance = sum(predictionerror**2)/(fullsize-2) - (sum(predictionerror)/(fullsize-2))**2
print variance

predictionerror = predictionerror - X[1:-1]
variance = sum(predictionerror**2)/(fullsize-2) - (sum(predictionerror)/(fullsize-2))**2
print variance