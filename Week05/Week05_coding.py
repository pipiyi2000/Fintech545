import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import functions as f

# Problem 2
df = pd.read_csv('problem1.csv')
df = df.values

# a. Normal distribution with an exponentially weighted variance
VaR_ew_N = f.simple_VaR(df,"EW_Normal",0.05,0.97)
ES_ew_N = f.simple_ES(df,"EW_Normal",0.05,0.97)
print(VaR_ew_N, ES_ew_N)

# b. MLE fitted T distribution
VaR_t = f.simple_VaR(df, "T", 0.05)
ES_t = f.simple_ES(df, "T", 0.05)
print(VaR_t, ES_t)

# c. Historic Simulation
rand = np.random.choice(df.shape[0], size=100000, replace=True)
sim_df = df[rand, :]
x_s = np.sort(sim_df, axis = 0)
n = 0.05 * len(x_s)
VaR_abs = -np.percentile(sim_df, 5)
VaR_diff_from_mean = - (- VaR_abs - np.mean(sim_df))
idn = int(np.floor(n))
ES_abs = -np.mean(x_s[0:idn])
ES_diff_from_mean = -(-ES_abs - np.mean(sim_df))
print(VaR_abs, VaR_diff_from_mean, ES_abs, ES_diff_from_mean)

# Problem 3
price = pd.read_csv('DailyPrices.csv')
current_price = price.iloc[-1]
df_returns = f.return_calculate(price)
print(df_returns)

portfolio = pd.read_csv('portfolio.csv')

df_price = pd.read_csv("DailyPrices.csv")
current_prices = df_price.iloc[-1]
df_ret = f.return_calculate(df_price, method = "DISCRETE")

# Calculate Portfolio A
port = "A"
nms = portfolio[portfolio["Portfolio"].isin(["A","B","C"]) if port == "Total" else portfolio["Portfolio"] == port]["Stock"].values

nms = np.append(nms,"SPY")
returns = df_ret[nms]

stocks = [nm for nm in nms if nm != "SPY"]

Portfolio = portfolio[portfolio["Portfolio"].isin(["A","B","C"]) if port == "Total" else portfolio["Portfolio"] == port][["Stock","Holding"]]

returns = returns - np.mean(returns, axis = 0)

# Model fit
fittedModels = {'SPY': f.fit_normal(returns['SPY'])}
for stock in stocks:
    fittedModels[stock] = f.fit_regression_t(returns[stock], returns['SPY'])[1]
    
# Construct Copula, and generate simulated data
U = pd.DataFrame({nm: fittedModels[nm].u for nm in nms})
R = U.corr(method='spearman')
N_Sim = 5000
simU = pd.DataFrame(stats.norm.cdf(f.simulate_pca(R, N_Sim)), columns=nms)

simulatedReturns = pd.DataFrame({'SPY': fittedModels['SPY'].eval(simU['SPY'])})
for stock in stocks:
    simulatedReturns[stock] = fittedModels[stock].eval(simulatedReturns["SPY"],simU[stock])

iteration = np.arange(1, N_Sim + 1)
values = pd.DataFrame([(stock, Portfolio[Portfolio["Stock"] == stock]["Holding"].values[0], iter) for stock in Portfolio["Stock"] for iter in iteration], columns=['stock', 'holding', 'iteration'])
values['currentValue'] = values.apply(lambda row: current_prices[row['stock']] * row['holding'], axis=1)
values['simulatedValue'] = values.apply(lambda row: row['currentValue'] * (1.0 + simulatedReturns.loc[row['iteration'] - 1, row['stock']]), axis=1)
values['pnl'] = values['simulatedValue'] - values['currentValue']

# Calculate VaR and ES
def calculate_risk_metrics(group):
    return pd.Series({
        'VaR95': f.VaR_ES(group['pnl'], alpha=0.05)[0],
        'ES95': f.VaR_ES(group['pnl'], alpha=0.05)[1],
        'VaR99': f.VaR_ES(group['pnl'], alpha=0.01)[0],
        'ES99': f.VaR_ES(group['pnl'], alpha=0.01)[1],
    })

stockRisk = values.groupby('stock').apply(calculate_risk_metrics).reset_index()
total_pnl_per_iteration = values.groupby('iteration')['pnl'].sum().reset_index(name='pnl')
totalRisk = calculate_risk_metrics(total_pnl_per_iteration).to_frame().T
totalRisk['stock'] = 'Total'
output = pd.concat([stockRisk, totalRisk], ignore_index=True)
print(output)

# Calculate Portfolio B
port = "B"
nms = portfolio[portfolio["Portfolio"].isin(["A","B","C"]) if port == "Total" else portfolio["Portfolio"] == port]["Stock"].values

nms = np.append(nms,"SPY")
returns = df_ret[nms]

stocks = [nm for nm in nms if nm != "SPY"]

Portfolio = portfolio[portfolio["Portfolio"].isin(["A","B","C"]) if port == "Total" else portfolio["Portfolio"] == port][["Stock","Holding"]]

returns = returns - np.mean(returns, axis = 0)

# Model fit
fittedModels = {'SPY': f.fit_normal(returns['SPY'])}
for stock in stocks:
    fittedModels[stock] = f.fit_regression_t(returns[stock], returns['SPY'])[1]
    
# Construct Copula, and generate simulated data
U = pd.DataFrame({nm: fittedModels[nm].u for nm in nms})
R = U.corr(method='spearman')
N_Sim = 5000
simU = pd.DataFrame(stats.norm.cdf(f.simulate_pca(R, N_Sim)), columns=nms)

simulatedReturns = pd.DataFrame({'SPY': fittedModels['SPY'].eval(simU['SPY'])})
for stock in stocks:
    simulatedReturns[stock] = fittedModels[stock].eval(simulatedReturns["SPY"],simU[stock])

iteration = np.arange(1, N_Sim + 1)
values = pd.DataFrame([(stock, Portfolio[Portfolio["Stock"] == stock]["Holding"].values[0], iter) for stock in Portfolio["Stock"] for iter in iteration], columns=['stock', 'holding', 'iteration'])
values['currentValue'] = values.apply(lambda row: current_prices[row['stock']] * row['holding'], axis=1)
values['simulatedValue'] = values.apply(lambda row: row['currentValue'] * (1.0 + simulatedReturns.loc[row['iteration'] - 1, row['stock']]), axis=1)
values['pnl'] = values['simulatedValue'] - values['currentValue']

# Calculate VaR and ES
def calculate_risk_metrics(group):
    return pd.Series({
        'VaR95': f.VaR_ES(group['pnl'], alpha=0.05)[0],
        'ES95': f.VaR_ES(group['pnl'], alpha=0.05)[1],
        'VaR99': f.VaR_ES(group['pnl'], alpha=0.01)[0],
        'ES99': f.VaR_ES(group['pnl'], alpha=0.01)[1],
    })

stockRisk = values.groupby('stock').apply(calculate_risk_metrics).reset_index()
total_pnl_per_iteration = values.groupby('iteration')['pnl'].sum().reset_index(name='pnl')
totalRisk = calculate_risk_metrics(total_pnl_per_iteration).to_frame().T
totalRisk['stock'] = 'Total'
outpout = pd.concat([stockRisk, totalRisk], ignore_index=True)
print(output)

# Calculate C
port = "C"
nms = portfolio[portfolio["Portfolio"].isin(["A","B","C"]) if port == "Total" else portfolio["Portfolio"] == port]["Stock"].values

nms = np.append(nms,"SPY")
returns = df_ret[nms]

stocks = [nm for nm in nms if nm != "SPY"]

Portfolio = portfolio[portfolio["Portfolio"].isin(["A","B","C"]) if port == "Total" else portfolio["Portfolio"] == port][["Stock","Holding"]]

returns = returns - np.mean(returns, axis = 0)

# Model fit
fittedModels = {}

# Fit a normal model to the market proxy 'SPY'
fittedModels['SPY'] = f.fit_normal(returns['SPY'])

# Fit normal models to individual stocks
for stock in stocks:
    fittedModels[stock] = f.fit_normal(returns[stock])
    
# Construct Copula, and generate simulated data
U = pd.DataFrame({nm: fittedModels[nm].u for nm in nms})
R = U.corr(method='spearman')
N_Sim = 5000
simU = pd.DataFrame(stats.norm.cdf(f.simulate_pca(R, N_Sim)), columns=nms)

simulatedReturns = pd.DataFrame(index=simU.index)
for nm in nms:
    simulatedReturns[nm] = fittedModels[nm].eval(simU[nm])

iteration = np.arange(1, N_Sim + 1)
values = pd.DataFrame([(stock, Portfolio[Portfolio["Stock"] == stock]["Holding"].values[0], iter) for stock in Portfolio["Stock"] for iter in iteration], columns=['stock', 'holding', 'iteration'])
values['currentValue'] = values.apply(lambda row: current_prices[row['stock']] * row['holding'], axis=1)
values['simulatedValue'] = values.apply(lambda row: row['currentValue'] * (1.0 + simulatedReturns.loc[row['iteration'] - 1, row['stock']]), axis=1)
values['pnl'] = values['simulatedValue'] - values['currentValue']

# Calculate VaR and ES
def calculate_risk_metrics(group):
    return pd.Series({
        'VaR95': f.VaR_ES(group['pnl'], alpha=0.05)[0],
        'ES95': f.VaR_ES(group['pnl'], alpha=0.05)[1],
        'VaR99': f.VaR_ES(group['pnl'], alpha=0.01)[0],
        'ES99': f.VaR_ES(group['pnl'], alpha=0.01)[1],
    })

stockRisk = values.groupby('stock').apply(calculate_risk_metrics).reset_index()
total_pnl_per_iteration = values.groupby('iteration')['pnl'].sum().reset_index(name='pnl')
totalRisk = calculate_risk_metrics(total_pnl_per_iteration).to_frame().T
totalRisk['stock'] = 'Total'
output = pd.concat([stockRisk, totalRisk], ignore_index=True)
print(output)