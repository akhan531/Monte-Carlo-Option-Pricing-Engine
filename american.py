import numpy as np
from scipy import stats
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt


# Uses machine learning for regression and optimal stopping
def longstaff_schwartz(S0, r, sigma, T, K, n_trials, n_timesteps, option_type, ml_model):
    S = np.zeros((n_trials, n_timesteps))
    S[:,0] = S0
    rng = np.random.default_rng(42)
    dW = rng.normal(0, 1, (n_trials, n_timesteps))
    dt = T / n_timesteps
    for t in range(1,n_timesteps):
        S_t = S[:,t-1] 
        S[:,t] = S_t + r*S_t*dt + sigma*S_t*dW[:,t-1]


    ST = S[:,n_timesteps-1]
    payoffs = np.zeros((n_trials, n_timesteps))

    if option_type == 'call':
        payoffs[:,n_timesteps-1] = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs[:,n_timesteps-1] = np.maximum(K - ST, 0)

    for t in range(n_timesteps-2,-1,-1):
        if option_type == 'call':
            itm_indices = np.where(S[:,t] > K)
            otm_indices = np.where(S[:,t] <= K)
        elif option_type == 'put':
            itm_indices = np.where(S[:,t] < K)
            otm_indices = np.where(S[:,t] >= K)
        if len(itm_indices[0]) == 0:
            payoffs[:, t] = payoffs[:, t+1] * np.exp(-r*dt)
            continue
        X = S[itm_indices, t].reshape(-1, 1)
        y = payoffs[itm_indices,t+1] * np.exp(-r*dt)
        match ml_model:
            case 'poly':
                model = make_pipeline(
                    PolynomialFeatures(degree=3),
                    LinearRegression(fit_intercept=False)
                )
                model.fit(X, y.ravel())
            case 'random forest':
                model = RandomForestRegressor(
                    n_estimators=200,  
                    max_depth=5,         
                    min_samples_leaf=10, 
                    random_state=42
                )
                model.fit(X, y.ravel())
            case 'xgboost':
                model = xgb.XGBRegressor()
                model.fit(X, y)
            case 'mlp':
                model = MLPRegressor(
                    hidden_layer_sizes=(64, 64),  
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=42
                )
                model.fit(X, y.ravel())
            case _:
                raise ValueError(f"Unknown model type: {ml_model}")


        future_val = model.predict(X).flatten()
        if option_type == 'call':
            current_val = np.maximum(S[itm_indices, t] - K, 0).flatten()
        elif option_type == 'put':
            current_val = np.maximum(K - S[itm_indices, t], 0).flatten()

        itm = itm_indices[0]  
        should_exercise = current_val > future_val

        payoffs[itm[should_exercise], t] = current_val[should_exercise]
        payoffs[itm[~should_exercise], t] = payoffs[itm[~should_exercise], t+1] * np.exp(-r*dt)
        payoffs[otm_indices, t] = payoffs[otm_indices, t+1] * np.exp(-r*dt)


    profit = np.mean(payoffs[:,0])

    SE = np.std(payoffs[:,0], ddof=1) / np.sqrt(n_trials)
    lower = profit-1.96*SE
    upper = profit+1.96*SE

    return profit, lower, upper

def compare_models(S0, r, sigma, T, K, n_trials, n_timesteps, option_type):
    models = ['poly', 'random forest', 'xgboost', 'mlp'];
    d = dict(zip(models,[longstaff_schwartz(S0, r, sigma, T, K, n_trials, n_timesteps, option_type, model) for model in models]))
    return d

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def binomial_tree(S0, r, sigma, T, K, n_timesteps, option_type):
    stock_prices = TreeNode(S0)
    option_prices = TreeNode()
    dt = T / n_timesteps
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d) / (u - d)

    def fillTree(sRoot, oRoot, depth):
        if depth == n_timesteps:
            if option_type == 'call':
                oRoot.val = max(sRoot.val - K, 0)
            elif option_type == 'put':
                oRoot.val = max(K - sRoot.val, 0)
            return
        sRoot.left = TreeNode(sRoot.val*d)
        sRoot.right = TreeNode(sRoot.val*u)
        oRoot.left = TreeNode()
        oRoot.right = TreeNode()
        fillTree(sRoot.left, oRoot.left, depth+1)
        fillTree(sRoot.right, oRoot.right, depth+1)
        if option_type == 'call':
            exercise_val = max(sRoot.val - K, 0)
        elif option_type == 'put':
            exercise_val = max(K - sRoot.val, 0)
        expected_val = ((p * oRoot.right.val) + ((1-p) * oRoot.left.val)) * np.exp(-r*dt)
        oRoot.val = max(exercise_val,expected_val);

    fillTree(stock_prices, option_prices, 0)

    return option_prices.val



    
    



S0 = 100
n_trials = 100
n_timesteps = 100
option_type = 'call'

test_T = [0.25, 0.5, 1]
test_sigma = [0.15, 0.25, 0.4]
test_moneyness = [-10, 0, 10]
test_r = [0.01,0.03,0.05]
test_models = ['poly', 'random forest', 'xgboost', 'mlp']

parameters = list(product(test_T, test_sigma, test_moneyness, test_r, test_models))

option_prices = pd.DataFrame(
    index = pd.MultiIndex.from_tuples(parameters, names=['T', 'sigma', 'moneyness', 'r', 'model']),
    columns=['Model Profit', 'Binomial Profit', 'Lower', 'Upper', 'Absolute Error', 'Relative Error']
)

for T, sigma, moneyness, r, _ in parameters:
    K = S0 * (1 + moneyness/100)  
    d = compare_models(S0, r, sigma, T, K, n_trials, n_timesteps, option_type)
    binomial_price = binomial_tree(S0, r, sigma, T, K, n_timesteps, option_type)
    for model in test_models:
        model_price = d[model][0]
        option_prices.loc[(T, sigma, moneyness, r, model), 'Model Profit'] = model_price
        option_prices.loc[(T, sigma, moneyness, r, model), 'Binomial Profit'] = binomial_price
        option_prices.loc[(T, sigma, moneyness, r, model), 'Lower'] = d[model][1]
        option_prices.loc[(T, sigma, moneyness, r, model), 'Upper'] = d[model][2]
        option_prices.loc[(T, sigma, moneyness, r, model), 'Absolute Error'] = abs(model_price-binomial_price)
        option_prices.loc[(T, sigma, moneyness, r, model), 'Relative Error'] = (model_price-binomial_price) / binomial_price * 100

model_metrics = pd.DataFrame(
    index = test_models,
    columns = ['RMSE', 'MAPE', 'poly p-value', 'random forest p-value', 'xgboost p-value', 'mlp p-value']
)

for model1 in test_models:
    abs_error_data1 = option_prices.xs(model1, level='model')['Absolute Error']
    rel_error_data1 = option_prices.xs(model1, level='model')['Relative Error']
    n = len(abs_error_data1)
    model_metrics.loc[model1, 'RMSE'] = np.sqrt(((abs_error_data1 ** 2).sum()) / n)
    model_metrics.loc[model1, 'MAPE'] = (abs(rel_error_data1).sum()) / n

    for model2 in test_models:
        abs_error_data2 = option_prices.xs(model2, level='model')['Absolute Error']
        t_stat, p_value = stats.ttest_rel(abs_error_data1, abs_error_data2)
        model_metrics.loc[model1, (model2 + ' p-value')] = p_value

    
