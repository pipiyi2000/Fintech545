import numpy as np
import pandas as pd
from scipy import stats
from bisect import bisect_left
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from scipy.stats import norm, t
from scipy import optimize
from scipy import integrate

# Problem 1
def covariance_complete_case(data):
    complete_data = data.dropna()
    return np.cov(complete_data.T, bias=False)

def correlation_complete_case(data):
    complete_data = data.dropna()
    return np.corrcoef(complete_data.T)

def covariance_pairwise(data):
    cov_matrix = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            valid_indices = data.iloc[:, [i, j]].dropna().index
            cov_matrix[i, j] = np.cov(data.iloc[valid_indices, i], data.iloc[valid_indices, j], bias=False)[0, 1]
    return cov_matrix

def correlation_pairwise(data):
    corr_matrix = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            valid_indices = data.iloc[:, [i, j]].dropna().index
            corr_matrix[i, j] = np.corrcoef(data.iloc[valid_indices, i], data.iloc[valid_indices, j])[0, 1]
    return corr_matrix

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

# 1.1 Covariance estimation techniques
def exp_w_cov(x, lambda_):
    if type(x) != np.ndarray:
        x = x.values
    m, n = x.shape
    w = np.empty(m)
    xm = np.mean(x, axis=0)
    x = (x - xm)

    w = (1 - lambda_) * lambda_ ** np.arange(m)[::-1]
    w /= np.sum(w)

    w = w.reshape(-1, 1)
    return (w * x).T @ x

def exp_w_corr(x, lambda_):
    cov = exp_w_cov(x, lambda_)
    invSD = np.diag(1.0 / np.sqrt(np.diag(cov)))
    corr = np.dot(invSD, cov).dot(invSD)
    return corr

def ewCovarCor(x, lambda_cov, lambda_corr):
    cov = exp_w_cov(x,lambda_cov)
    corr = exp_w_corr(x,lambda_corr)
    invSD = np.diag(np.sqrt(np.diag(cov)))
    out = invSD @ corr @ invSD
    return out

# 1.2 Non PSD fixes for correlation matrices          
def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    invSD = None
    out = a.copy()

    # Calculate the correlation matrix if we got a covariance matrix
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    # Eigen decomposition, update the eigenvalues, and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (vecs ** 2 @ vals[:, np.newaxis])
    T = np.diag(np.sqrt(T.flatten()))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out

def _getAplus(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < 0] = 0
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def _getPS(A, W):
    W05 = np.sqrt(W)
    iW = np.linalg.inv(W05)
    return iW @ _getAplus(W05 @ A @ W05) @ iW

def wgtNorm(A, W):
    W05 = np.sqrt(W)
    return np.sum((W05 @ A @ W05)**2)

def higham_nearest_psd(pc, W=None, epsilon=1e-9, maxIter=100, tol=1e-9):
    n = pc.shape[0]
    if W is None:
        W = np.diag(np.ones(n))

    Yk = np.copy(pc)
    norml = np.inf
    i = 1

    while i <= maxIter:
        Rk = Yk
        # Ps Update
        Xk = _getPS(Rk, W)
        # Get Norm
        norm = wgtNorm(Yk - pc, W)
        # Smallest Eigenvalue
        minEigVal = np.min(np.linalg.eigvalsh(Yk))

        if abs(norm - norml) < tol and minEigVal > -epsilon:
            # Norm converged and matrix is at least PSD
            break

        norml = norm
        Yk = Xk
        i += 1

    return Yk

def chol_psd(cov_matrix):
    n = cov_matrix.shape[0]
    root = np.zeros_like(cov_matrix)
    for j in range(n):
        s = 0.0
        if j > 0:
            # calculate dot product of the preceeding row values
            s = np.dot(root[j, :j], root[j, :j])
        temp = cov_matrix[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)
        if root[j, j] == 0.0:
            # set the column to 0 if we have an eigenvalue of 0
            root[j + 1:, j] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (cov_matrix[i, j] - s) * ir
    return root

# 1.3 Simulation Methods
def simulate_normal(N, cov, mean=None, seed=1234):
    n = cov.shape[0]
    if cov.shape[1] != n:
        raise ValueError(f"Covariance matrix is not square ({n},{cov.shape[1]})")

    if mean is None:
        mean = np.zeros(n)
    elif mean.shape[0] != n:
        raise ValueError(f"Mean ({mean.shape[0]}) is not the size of cov ({n},{n})")


    l = chol_psd(cov) 

    # Generate needed random standard normals
    np.random.seed(seed)
    out = np.random.standard_normal((N, n))

    # Apply the Cholesky root to the standard normals
    out = np.dot(out, l.T)

    # Add the mean
    out += mean

    return out

def simulate_pca(a, nsim, pctExp=1, mean=None, seed=1234):
    n = a.shape[0]

    if mean is None:
        mean = np.zeros(n)
    elif mean.shape[0] != n:
        raise ValueError(f"Mean size {mean.shape[0]} does not match covariance size {n}.")

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)
    vals = np.real(vals)
    vecs = np.real(vecs)
    # Sort eigenvalues and eigenvectors
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Calculate total variance
    tv = np.sum(vals)

    # Select principal components based on pctExp
    cum_var_exp = np.cumsum(vals) / tv
    if pctExp < 1:
        n_components = np.searchsorted(cum_var_exp, pctExp) + 1
        vals = vals[:n_components]
        vecs = vecs[:, :n_components]
    else:
        n_components = n
    # Construct principal component matrix
    B = vecs @ np.diag(np.sqrt(vals))

    # Generate random samples
    np.random.seed(seed)
    r = np.random.randn(n_components, nsim)
    out = (B @ r).T

    # Add the mean
    out += mean

    return out

# return calculate
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

# fitted model
class FittedModel:
    def __init__(self, beta, error_model, eval_func, errors, u):
        self.beta = beta
        self.error_model = error_model
        self.eval = eval_func
        self.errors = errors
        self.u = u

def fit_normal(x):
    m = np.mean(x)
    s = np.std(x, ddof = 1)
    
    error_model = norm(m, s)
    
    errors = x - m
    u = error_model.cdf(x)
    
    def eval_u(u):
        return error_model.ppf(u)
    
    return FittedModel(None, error_model, eval_u, errors, u)

def fit_t(x):
    params = t.fit(x)
    df, loc, scale = params
    error_model = t(df=df, loc=loc, scale=scale)
    
    errors = x - loc
    
    u = error_model.cdf(x)
    
    def eval_u(u):
        return error_model.ppf(u)
    
    fit_model = FittedModel(None, error_model, eval_u, errors, u)
    opt_para = loc, scale, df
    return np.array(opt_para), fit_model

def general_t_ll(mu, s, nu, x):
    td = stats.t(df=nu, loc=mu, scale=s)
    return np.sum(np.log(td.pdf(x)))

def fit_regression_t(y, x):
    if len(x.shape) == 1:
        x = x.values.reshape(-1,1)
    if type(y) != np.ndarray:
        y = y.values
    n = x.shape[0]
    X = np.hstack((np.ones((n, 1)), x))
    nB = X.shape[1]

    b_start = np.linalg.inv(X.T @ X) @ X.T @ y
    e = y - X @ b_start
    start_m = np.mean(e)
    start_nu = 6.0 / stats.kurtosis(e, fisher=False) + 4
    start_s = np.sqrt(np.var(e) * (start_nu - 2) / start_nu)

    def objective(params):
        m, s, nu, *B = params
        xm = y - X @ np.array(B)
        return -general_t_ll(m, s, nu, xm)

    initial_params = [start_m, start_s, start_nu] + b_start.tolist()
    bounds = [(None, None), (1e-6, None), (2.0001, None)] + [(None, None)] * nB
    result = optimize.minimize(objective, initial_params, bounds=bounds)

    m, s, nu, *beta = result.x
    errorModel = t(df=nu, loc=m, scale=s)

    def eval_model(x, u):
        if len(x.shape) == 1:
            x = x.values.reshape(-1,1)
        n = x.shape[0]
        _temp = np.hstack((np.ones((n, 1)), x))
        return _temp @ np.array(beta) + errorModel.ppf(u)

    errors = y - eval_model(x, np.full(x.shape[0], 0.5))
    u = errorModel.cdf(errors)
    opt_para = result.x
    fit_model = FittedModel(beta, errorModel, eval_model, errors, u)
    return np.array(opt_para), fit_model

# 1.4 VaR calculation methods
def VaR_cal(method, ret, PV, Asset_value, holdings, name, current_prices, alpha):

    # Calcualte Covariance Matrix and Portfiolio Volaitility
    if method == "Normal":
        # R_gradients also equal to weights
        R_gradients = np.array(Asset_value) / PV
        Sigma = np.cov(ret, rowvar=False)
        p_sig = np.sqrt(np.dot(R_gradients.T, np.dot(Sigma, R_gradients)))
        VaR = (-PV) * norm.ppf(alpha) * p_sig
    
    elif method == "MLE_T":
        params = stats.t.fit(ret)
        df, loc, scale = params
        VaR = (-PV) * stats.t.ppf(alpha, df, loc, scale)
    
    elif method == "Historical":
        rand_indices = np.random.choice(ret.shape[0], size=10000, replace=True)
        sim_ret = ret.values[rand_indices, :]
        sim_price = current_prices.values * (1 + sim_ret)
        vHoldings = np.array([holdings[nm] for nm in name])
        pVals = sim_price @ vHoldings
        VaR = PV - np.percentile(pVals, alpha * 100)
    return VaR

def simple_VaR(rets, dist, alpha = 0.05, lbda = 0.97):
    if type(rets) != np.ndarray:
        rets = rets.values.reshape(-1,1)
    if dist == "Normal":
        fitted_model = fit_normal(rets) 
        VaR_abs =  -norm.ppf(alpha, fitted_model.error_model.mean(), fitted_model.error_model.std())
        VaR_diff_from_mean = -(-VaR_abs - fitted_model.error_model.mean())
        return np.array([VaR_abs, VaR_diff_from_mean])
    elif dist == "EW_Normal":
        std = np.sqrt(exp_w_cov(rets,lbda))
        VaR_abs =  -norm.ppf(alpha, np.mean(rets), std)
        VaR_diff_from_mean = -(-VaR_abs - np.mean(rets))
        return np.array([VaR_abs, VaR_diff_from_mean]).reshape(-1)
    elif dist == "T":
        opt_para, fitted_model = fit_t(rets)
        VaR_abs = -t.ppf(alpha, df = opt_para[2], loc = opt_para[0], scale = opt_para[1])
        VaR_diff_from_mean = -(-VaR_abs - opt_para[0])
        return np.array([VaR_abs, VaR_diff_from_mean])

def simple_VaR_sim(rets, dist, alpha = 0.05, N = 100000):
    if type(rets) != np.ndarray:
        rets = rets.values
    if dist == "Normal":
        fitted_model = fit_normal(rets)
        rand_num = norm.rvs(fitted_model.error_model.mean(),fitted_model.error_model.std(), size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        VaR_abs = -(xs[iup] + xs[idn]) / 2
        VaR_diff_from_mean = -(-VaR_abs - np.mean(xs))
        return np.array([VaR_abs, VaR_diff_from_mean])
    elif dist == "T":
        opt_para, fit_model = fit_t(rets)
        rand_num = t.rvs(df = opt_para[2], loc = opt_para[0], scale = opt_para[1], size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        VaR_abs = -(xs[iup] + xs[idn]) / 2
        VaR_diff_from_mean = -(-VaR_abs - np.mean(xs))
        return np.array([VaR_abs, VaR_diff_from_mean]) 
        
def simple_ES(rets, dist, alpha = 0.05, lbda = 0.97):
    if type(rets) != np.ndarray:
        rets = rets.values.reshape(-1,1)
    if dist == "Normal":
        VaR_abs = simple_VaR(rets, dist, alpha)[0]
        fitted_model = fit_normal(rets)
        def integrand(x):
            return x * norm.pdf(x,fitted_model.error_model.mean(), fitted_model.error_model.std())
        integral_abs, error = integrate.quad(integrand, -np.inf, -VaR_abs)
        ES_abs = - integral_abs / alpha
        ES_diff_from_mean = -(-ES_abs-fitted_model.error_model.mean())
        return np.array([ES_abs, ES_diff_from_mean])
    
    elif dist == "EW_Normal":
        VaR_abs = simple_VaR(rets, dist, alpha, lbda)[0]
        def integrand(x):
            std = np.sqrt(exp_w_cov(rets, lbda)[0])
            return x * norm.pdf(x,np.mean(rets), std)
        integral_abs, error = integrate.quad(integrand, -np.inf, -VaR_abs)
        ES_abs = - integral_abs / alpha
        ES_diff_from_mean = -(-ES_abs-np.mean(rets))
        return np.array([ES_abs, ES_diff_from_mean])
    
    elif dist == "T":
        VaR_abs = simple_VaR(rets, dist, alpha)[0]
        opt_para, fitted_model = fit_t(rets)
        def integrand(x):
            return x * t.pdf(x,df = opt_para[2], loc = opt_para[0], scale = opt_para[1])
        integral_abs, error = integrate.quad(integrand, -np.inf, -VaR_abs)

        ES_abs = - integral_abs / alpha
        ES_diff_from_mean = -(-ES_abs-opt_para[0])
        return np.array([ES_abs, ES_diff_from_mean])

def simple_ES_sim(rets, dist, alpha = 0.05, N = 1000000):
    if type(rets) != np.ndarray:
        rets = rets.values
    if dist == "Normal":
        fitted_model = fit_normal(rets)
        rand_num = norm.rvs(fitted_model.error_model.mean(),fitted_model.error_model.std(), size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        ES_abs = -np.mean(xs[0:idn])
        ES_diff_from_mean = -(-ES_abs - np.mean(xs))
        return np.array([ES_abs, ES_diff_from_mean])
    elif dist == "T":
        opt_para, fit_model = fit_t(rets)
        rand_num = t.rvs(df = opt_para[2], loc = opt_para[0], scale = opt_para[1], size = N)
        xs = np.sort(rand_num)
        n = alpha * len(xs)
        iup = int(np.ceil(n))
        idn = int(np.floor(n))
        ES_abs = -np.mean(xs[0:idn])
        ES_diff_from_mean = -(-ES_abs - np.mean(xs))
        return np.array([ES_abs, ES_diff_from_mean])

# VaR/ES calculation
def VaR_ES(x, alpha=0.05):
    xs = np.sort(x)
    n = alpha * len(xs)
    iup = int(np.ceil(n))
    idn = int(np.floor(n))
    VaR = (xs[iup] + xs[idn]) / 2
    ES = np.mean(xs[0:idn])
    return -VaR, -ES