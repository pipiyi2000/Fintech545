import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t, ttest_1samp, multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

#problem 1.a
def four_moments(sample):
    n = len(sample)
    #mean
    u_hat = np.sum(sample)/n
    #variance
    sigma2_hat = np.sum((sample - u_hat)**2) / n
    #skew
    skew_hat = np.sum(((sample - u_hat) / np.sqrt(sigma2_hat))**3) / n
    #kurtosis
    excessKurt_hat = np.sum((sample - u_hat)**4) / n / sigma2_hat**2 - 3
    return u_hat, sigma2_hat, skew_hat, excessKurt_hat

data = np.array([])
#read datafile
with open('problem1.csv', 'r') as file:
    next(file)
    for line in file:
        #collect data from file
        data = np.append(data, float(line.strip()))
print("The four moments are: ", four_moments(data))

#problem 1.b
moment_mean = np.mean(data)
moment_variance = np.var(data)
moment_skewness = stats.skew(data)
moment_kurtosis = stats.kurtosis(data)
print("The four moments are: ", moment_mean, moment_variance, moment_skewness, moment_kurtosis)

#problem 1.c
#create samples for testing
samples = 100000
sample_size = 10000
kurts = np.empty(samples)
for i in range(samples):
    sample = np.random.normal(0, 1, sample_size)
    kurts[i] = stats.kurtosis(sample, fisher=False)
#compare p-value for testing
t_statistic = np.mean(kurts) / np.sqrt(np.var(kurts) / samples)
p_value = 2 * (1 - t.cdf(abs(t_statistic), df=samples - 1))
ttest = ttest_1samp(kurts, 3)
p_value_2 = ttest.pvalue
print(p_value, p_value_2)
print("Match the stats package test?:", np.isclose(p_value, p_value_2))

#prblem 2.a
#read datafile and extract x, y
data = pd.read_csv('problem2.csv')
data['one'] = 1
x = data['x']
X = data[['one', 'x']]
y = data['y']
#fit OLS
OLS_model = sm.OLS(y, X).fit()
print(OLS_model.summary())
residuals = OLS_model.resid
std = residuals.std()
print(OLS_model.params, OLS_model.bse, std)

#fit MLE given normality
n = len(y)
def MLE_normality(parameters,x, y):
    beta0, beta1, sd = parameters
    expect_y = beta0 + x * beta1
    ll = -1 * np.sum(stats.norm.logpdf(y, expect_y, sd))
    return ll
MLE_norm = minimize(MLE_normality, np.array([1, 1, 1]), args=(x, y))
print(MLE_norm)
print(MLE_norm.x)

#problem 2.b
#fit MLE given t-distribution
def MLE_tdistribution(parameters, x, y):
    sd = parameters[0]
    df = parameters[1]
    beta0 = parameters[2]
    beta1 = parameters[3]
    error = y - beta0 - x * beta1
    ll = -1 * np.sum(stats.t.logpdf(error, df, scale=sd))
    return ll
MLE_t = minimize(MLE_tdistribution, np.array([1, 1, 0, 0]), args=(x, y))
print(MLE_t)
print(MLE_t.x)
#compare two models based on AIC and BIC
AIC_norm = 2 * 3 - 2 * (-MLE_norm.fun)
AIC_t = 2 * 4 - 2 * (-MLE_t.fun)
print("AIC_norm < AIC_t:", AIC_norm < AIC_t)
BIC_norm = np.log(n) * 3 - 2 * (-MLE_norm.fun)
BIC_t = np.log(n) * 4 - 2 * (-MLE_t.fun)
print("BIC_norm < BIC_t:", BIC_norm < BIC_t)

#problem 2.c
#read data files
data_2c = pd.read_csv('problem2_x.csv')
data_x1 = pd.read_csv('problem2_x1.csv')
mean = data_2c.mean()
cov = data_2c.cov()
x_1 = data_x1['x1']
#create model
multi_norm = multivariate_normal(mean=mean, cov=cov)
def condi_dis(x1, mean, cov):
    mean_1, mean_2 = mean
    cov_xx, cov_xy, cov_yx, cov_yy = cov.values.flatten()
    condi_mean = mean_2 + cov_yx / cov_xx * (x1 - mean_1)
    condi_cov = cov_yy - cov_yx * cov_xy / cov_xx
    return condi_mean, condi_cov
#store bounds
condi_meanlist = []
lower_interval = []
upper_interval = []
#get expected value bounds
for x1 in x_1:
    condi_mean, condi_cov = condi_dis(x1, mean, cov)
    condi_meanlist.append(condi_mean)
    interval = 1.96 * np.sqrt(condi_cov)
    lower_interval.append(condi_mean - interval)
    upper_interval.append(condi_mean + interval)
#plot the graph 
plt.figure(figsize=(10, 6))
plt.plot(x_1, condi_meanlist, color='red', label='Expected x2')
plt.fill_between(x_1, lower_interval, upper_interval, color='gray', alpha=0.5, label="95% CI")
plt.xlabel('x1')
plt.ylabel('Explected x2')
plt.legend()
plt.show()

#problem 3
#read data file
data_3 = pd.read_csv('problem3.csv')
x = data_3['x']
#AR1 - AR3
ar_1 = ARIMA(x, order=(1, 0, 0)).fit()
ar_2 = ARIMA(x, order=(2, 0, 0)).fit()
ar_3 = ARIMA(x, order=(3, 0, 0)).fit()
#MA1 - MA3
ma_1 = ARIMA(x, order=(0, 0, 1)).fit()
ma_2 = ARIMA(x, order=(0, 0, 2)).fit()
ma_3 = ARIMA(x, order=(0, 0, 3)).fit()
#find the best model through AIC
models = [ar_1, ar_2, ar_3, ma_1, ma_2, ma_3]
best = None
for model in models:
    aic = model.aic
    print(aic)
    if aic < float('inf'):
        best = model
#plot the models
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
axes = axes.flatten()
axes[0].plot(x, label='Original')
axes[0].plot(ar_1.predict(), label='AR(1)')
axes[0].set_title('AR(1) Model')
axes[0].legend()

axes[1].plot(x, label='Original')
axes[1].plot(ar_2.predict(), label='AR(2)')
axes[1].set_title('AR(2) Model')
axes[1].legend()

axes[2].plot(x, label='Original')
axes[2].plot(ar_3.predict(), label='AR(3)')
axes[2].set_title('AR(3) Model')
axes[2].legend()

axes[3].plot(x, label='Original')
axes[3].plot(ma_1.predict(), label='MA(1)')
axes[3].set_title('MA(1) Model')
axes[3].legend()

axes[4].plot(x, label='Original')
axes[4].plot(ma_2.predict(), label='MA(2)')
axes[4].set_title('MA(2) Model')
axes[4].legend()

axes[5].plot(x, label='Original')
axes[5].plot(ma_3.predict(), label='MA(3)')
axes[5].set_title('MA(3) Model')
axes[5].legend()

plt.tight_layout()
plt.show()