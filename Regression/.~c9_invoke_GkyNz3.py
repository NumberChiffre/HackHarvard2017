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

fullsize = 46
windowsize = 5

def autocorr(x):
    nn = len(x)
    variance = x.var()
    x = x-x.mean()
    r = n.correlate(x, x, "full")[-nn:]
    assert n.allclose(r, n.array([(x[:nn-k]*x[-(nn-k):]).sum() for k in range(nn)]))
    result = r/(variance*(n.arange(nn, 0, -1)))
    return result

X = n.array([-1.252762909,
-0.693147151,
-0.693147151,
-1.504077332,
0,
-0.966014155,
-1.098612155,
-0.060624618,
-0.934309182,
0.628608601,
0.459532299,
0.606135728,
0.773189798,
0.139761936,
1.098611622,
1.299282742,
-0.916290582,
0.287681989,
1.84582641,
1.704747683,
2.014902587,
0.84729767,
-0.916290432,
0,
1.098611955,
0.200670675,
-0.356674901,
-0.587786576,
2.442346579,
0.773189798,
0.117783022,
0.182321523,
0.762139976,
0.84729767,
-0.356674901,
0,
0,
-0.887303111,
-0.18232154,
1.098612122,
0.154150656,
0.693147056,
0.84729767,
0.773189798,
0.367724746,
2.302584193
])

Y = n.array([-0.049370501,
0.044082282,
-0.004250803,
0.028348355,
0.013708234,
-0.086330873,
0.005919366,
-0.021249635,
-0.036451379,
0.028507323,
0.043840601,
-0.034390522,
0.020481073,
-0.074193036,
0.014972699,
-0.004311196,
-0.028687226,
0.009253737,
0.024526546,
0.006621251,
0.009273637,
0.001537279,
-0.039958312,
0.007961825,
-0.009962226,
-0.060249242,
-0.108626868,
-0.050569841,
-0.026277409,
0.111791406,
0.040821995,
0.014401296,
0.145697124,
0.017449858,
0.039687748,
-0.021091041,
-0.005070637,
-0.072656593,
-0.005481611,
0.059076044,
0.007374665,
-0.01891397,
0.02074917,
0.048667922,
-0.046836084,
0.170847982
])

data2d = n.array([Y,X])
sizedata = n.transpose(data2d)
spread = [el[1] - el[0] for el in sizedata]
#for l in sizedata:
#    print l
#print sizedata.shape
from statsmodels.tsa.api import VAR, DynamicVAR
vartest = VAR(sizedata)
gtest = grangercausalitytests(sizedata,maxlag=14, addconst=True, verbose=True)
adftest = adfuller(spread,regression='ct',regresults=True)
adfX, adfY = adfuller(X,regression='ct',regresults=True), adfuller(Y,regression='ct', regresults=True)
varresult = vartest.fit(10)
print varresult.summary()
print vartest.select_order(10)
print varresult.test_causality(vartest, sizedata,kind='f')
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
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

