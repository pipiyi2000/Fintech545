import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
from collections import defaultdict

# Problem 1
n_simu = 1000
Pt_1 = 100
s = 0.1
r_t = np.random.normal(0, s, n_simu)

#classical model
C_model = Pt_1 + r_t
expected_C = np.mean(C_model)
sd_C = np.std(C_model)

#arithmetic model
A_model = Pt_1 * (1 + r_t)
expected_A = np.mean(A_model)
sd_A = np.std(A_model)

#log model
L_model = Pt_1 * np.exp(r_t)
expected_L = np.mean(L_model)
sd_L = np.std(L_model)

#print results
print("After 1000 simulations, the mean expected value of the classical model is:", expected_C, ", and the mean standard deviation is ", sd_C)
print("After 1000 simulations, the mean expected value of the arithmetic model is:", expected_A, ", and the mean standard deviation is ", sd_A)
print("After 1000 simulations, the mean expected value of the log model is:", expected_L, ", and the mean standard deviation is ", sd_L)

# Problem 2.1
def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    if dateColumn not in prices.columns:
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {prices.columns}")
    
    #Select the columns excluding the date column
    vars = [col for col in prices.columns if col != dateColumn]
    nVars = len(vars)
    
    #Extract the prices matrix
    p = prices[vars].values
    n, m = p.shape
    p2 = np.empty((n-1, m))
    
    #Calculate the price ratios or returns
    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]
    
    #Adjust the returns based on the selected method
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
    
    #Prepare the output DataFrame
    dates = prices[dateColumn].iloc[1:].values
    out = pd.DataFrame({dateColumn: dates})
    for i, var in enumerate(vars):
        out[var] = p2[:, i]
    
    return out

#Read Datafile
prices = pd.read_csv('DailyPrices.csv')
returns = return_calculate(prices)
print(returns)

# Problem 2.2
meta = returns['META'] - returns['META'].mean()
print(meta)

# Problem 2.3
#VaR using normal distribution
returns_mean = meta.mean()
returns_sd = meta.std()
normal_returns = np.random.normal(returns_mean, returns_sd, 10000)
normal_returns.sort()
VaR_norm = -np.percentile(normal_returns, 5)
print(VaR_norm)

#VaR using normal distribution with an Exponentially Weighted variance
def ewma_VaR(returns, lambda_=0.94, confidence_level=0.05):
    ewma_sd = returns.ewm(alpha=1 - lambda_).std().iloc[-1]
    VaR = -stats.norm.ppf(confidence_level) * ewma_sd
    return VaR
VaR_ewma_normal = ewma_VaR(meta)
print(VaR_ewma_normal)

#VaR using MLE fitted T distribution
def MLE_VaR(returns, confidence_level=0.05):
    df, loc, scale = stats.t.fit(returns)
    VaR = -stats.t.ppf(confidence_level, df, loc=loc, scale=scale)
    return VaR
VaR_MLE = MLE_VaR(meta)
print(VaR_MLE)

#VaR using fitted AR(1) model
def AR1_VaR(returns, confidence_level=0.05, forecast_horizon=1):
    AR1_model = AutoReg(returns, lags=1).fit()
    residual_sd = AR1_model.resid.std()
    VaR = -stats.norm.ppf(confidence_level) * residual_sd
    VaR = VaR * np.sqrt(forecast_horizon)
    return VaR
VaR_AR1 = AR1_VaR(meta)
print(VaR_AR1)

#VaR using historic simulation
def historic_VaR(returns, confidence_level=0.05):
    sorted_returns = np.sort(returns)
    index = int(np.ceil(confidence_level * len(sorted_returns)))
    VaR = -sorted_returns[index]
    return VaR
VaR_historic = historic_VaR(meta)
print(VaR_historic)

# Problem 3.1
prices = pd.read_csv('DailyPrices.csv')
returns = return_calculate(prices)

#read portfolio datafile and separate three portfolios
portfolio = pd.read_csv('portfolio.csv')
A_holdings = {}
B_holdings = {}
C_holdings = {}

for index, row in portfolio.iterrows():
    if row['Portfolio'] == 'A':
        stock = row['Stock']
        holding = row['Holding']
        A_holdings[stock] = holding
    if row['Portfolio'] == 'B':
        stock = row['Stock']
        holding = row['Holding']
        B_holdings[stock] = holding
    if row['Portfolio'] == 'C':
        stock = row['Stock']
        holding = row['Holding']
        C_holdings[stock] = holding

#calculate VaR of separe portfolios
def ewma_portfolio(returns, prices, holdings, lambda_=0.94, confidence_level=0.05):
    nm = set(prices.columns) & set(holdings.keys())
    currentprices = prices.iloc[-1][list(nm)]
    returns = returns[list(nm)]
    PV = sum(holdings[stock] * currentprices[stock] for stock in nm)
    weights = np.array([holdings[stock] * currentprices[stock] / PV for stock in nm])
    ewma_cov = returns.ewm(span=(2/(1-lambda_))-1).cov(pairwise=True).iloc[-len(returns.columns):]
    portfolio_variance = np.dot(weights.T, np.dot(ewma_cov, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    VaR = -stats.norm.ppf(confidence_level) * portfolio_std * PV
    return VaR, PV
print(ewma_portfolio(returns, prices, A_holdings))
print(ewma_portfolio(returns, prices, B_holdings))
print(ewma_portfolio(returns, prices, C_holdings))

#calculate VaR of the aggregated portfolio
aggregated_holdings = defaultdict(int)
for holding in [A_holdings, B_holdings, C_holdings]:
    for stock, quantity in holding.items():
        aggregated_holdings[stock] += quantity
print(ewma_portfolio(returns, prices, aggregated_holdings))

# Problem 3.2
def historic_portfolio(returns, prices, holdings, confidence_level=0.05):
    nm = list(set(returns.columns).intersection(set(holdings.keys())))
    filtered_returns = returns[nm]
    current_prices = prices.iloc[-1][nm]
    filtered_current_prices = current_prices[nm]
    PV = sum(holdings[stock] * current_prices[stock] for stock in nm)
    simulated_prices = (1 + filtered_returns).multiply(filtered_current_prices, axis='columns')
    vHoldings = np.array([holdings[s] for s in nm])
    portfolio_values = simulated_prices.dot(vHoldings)
    portfolio_values.sort_values(inplace=True)
    index_at_risk = int(confidence_level * len(portfolio_values))
    VaR_historical = PV - portfolio_values.iloc[index_at_risk]
    return VaR_historical
print(historic_portfolio(returns, prices, A_holdings))
print(historic_portfolio(returns, prices, B_holdings))
print(historic_portfolio(returns, prices, C_holdings))
print(historic_portfolio(returns, prices, aggregated_holdings))