import numpy as np
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from itertools import product
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# Uses machine learning for regression and optimal stopping
def longstaff_schwartz(S0, r, sigma, T, K, n_trials, n_timesteps, option_type, ml_model):
    """
    Core Longstaff-Schwartz implementation with ML models
    """
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
                model = RandomForestRegressor()
                model.fit(X, y.ravel())
            case 'xgboost':
                model = xgb.XGBRegressor()
                model.fit(X, y)
            case 'mlp':
                model = MLPRegressor()
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
    models = ['poly', 'random forest', 'xgboost', 'mlp']
    results = {}
    for model in models:
        price, lower, upper = longstaff_schwartz(S0, r, sigma, T, K, n_trials, n_timesteps, option_type, model)
        results[model] = (price, lower, upper)
    return results

def binomial_tree(S0, r, sigma, T, K, n_timesteps, option_type):
    stock_prices = [[S0]]
    dt = T / n_timesteps
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d) / (u - d)

    for t in range(1,n_timesteps):
        new_prices = [ele*d for ele in stock_prices[-1]] + [S0* (u**t)]
        stock_prices.append(new_prices)

    if option_type == 'call':
        option_prices = [max(ele - K, 0) for ele in stock_prices[-1]]
    elif option_type == 'put':
        option_prices = [max(K - ele, 0) for ele in stock_prices[-1]]

    for t in range(n_timesteps-2,-1,-1):
        current_prices = [0]*(t+1)
        future_prices = option_prices
        for i in range(t+1):
            if option_type == 'call':
                exercise_val = max(stock_prices[t][i] - K, 0)
            elif option_type == 'put':
                exercise_val = max(K - stock_prices[t][i], 0)
            expected_val = ((p * future_prices[i+1]) + ((1-p) * future_prices[i])) * np.exp(-r*dt)
            current_prices[i] = max(exercise_val,expected_val);
        option_prices = current_prices


    return option_prices[0]






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
no_model_parameters = list(product(test_T, test_sigma, test_moneyness, test_r))

'''option_prices = pd.DataFrame(
    columns=['T', 'sigma', 'moneyness', 'r', 'model', 'Model Price', 'Binomial Price', 'Lower', 'Upper', 'Absolute Error', 'Relative Error']
)


for T, sigma, moneyness, r in no_model_parameters:
    K = S0 * (1 + moneyness/100)  
    d = compare_models(S0, r, sigma, T, K, n_trials, n_timesteps, option_type)
    binomial_price = binomial_tree(S0, r, sigma, T, K, n_timesteps, option_type)
    for model in test_models:
        print(T, sigma, moneyness, r, model)
        model_price = d[model][0]
        lower = d[model][1]
        upper = d[model][2]
        abs_error = abs(model_price-binomial_price)
        rel_error = (model_price-binomial_price) / binomial_price * 100
        option_prices.loc[len(option_prices)] = [T, sigma, moneyness, r, model, model_price, binomial_price, lower, upper, abs_error, rel_error]

option_prices.to_csv('option_prices.csv')
'''
option_prices = pd.read_csv('option_prices.csv')


model_metrics = pd.DataFrame(
    index = test_models,
    columns = ['RMSE', 'MAPE', 'poly p-value', 'random forest p-value', 'xgboost p-value', 'mlp p-value']
)

for model1 in test_models:
    abs_error_data1 = option_prices[option_prices['model'] == model1]['Absolute Error']
    rel_error_data1 = option_prices[option_prices['model'] == model1]['Relative Error']
    n = len(abs_error_data1)
    model_metrics.loc[model1, 'RMSE'] = np.sqrt(((abs_error_data1 ** 2).sum()) / n)
    model_metrics.loc[model1, 'MAPE'] = (abs(rel_error_data1).sum()) / n

    for model2 in test_models:
        abs_error_data2 = option_prices[option_prices['model'] == model2]['Absolute Error']
        t_stat, p_value = stats.ttest_rel(abs_error_data1, abs_error_data2)
        model_metrics.loc[model1, (model2 + ' p-value')] = p_value

#RMSE Comparison
plt.bar(model_metrics.index, model_metrics['RMSE'])
plt.title('RMSE for each model')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()  

#MAPE Comparison
plt.bar(model_metrics.index, model_metrics['MAPE'])
plt.title('MAPE for each model')
plt.xlabel('Model')
plt.ylabel('MAPE')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show() 



#Performance by T
for model in test_models:
    x = test_T
    abs_error_data = [option_prices[(option_prices['model'] == model) & (option_prices['T'] == T)]['Absolute Error'] for T in test_T]
    y = [np.sqrt(((ele ** 2).sum()) / len(ele)) for ele in abs_error_data]
    plt.plot(x, y, label=model, marker='o') 
plt.xlabel('Time to Expiration')
plt.ylabel('RMSE')
plt.title('Model performance across different time to expirations')
plt.legend() 
plt.grid(True)  
plt.show()

#Performance by sigma
for model in test_models:
    x = test_sigma
    abs_error_data = [option_prices[(option_prices['model'] == model) & (option_prices['sigma'] == sigma)]['Absolute Error'] for sigma in test_sigma]
    y = [np.sqrt(((ele ** 2).sum()) / len(ele)) for ele in abs_error_data]
    plt.plot(x, y, label=model, marker='o')  
plt.xlabel('Volatility')
plt.ylabel('RMSE')
plt.title('Model performance across different volatilities')
plt.legend()  
plt.grid(True) 
plt.show()

#Performance by moneyness
for model in test_models:
    x = [S0 * (1 + moneyness/100) for moneyness in test_moneyness]
    abs_error_data = [option_prices[(option_prices['model'] == model) & (option_prices['moneyness'] == moneyness)]['Absolute Error'] for moneyness in test_moneyness]
    y = [np.sqrt(((ele ** 2).sum()) / len(ele)) for ele in abs_error_data]
    plt.plot(x, y, label=model, marker='o')  
plt.xlabel('Strike Price')
plt.ylabel('RMSE')
plt.title('Model performance across different strike prices w S0 = 100')
plt.legend() 
plt.grid(True)  
plt.show()

#Performance by r
for model in test_models:
    x = test_r
    abs_error_data = [option_prices[(option_prices['model'] == model) & (option_prices['r'] == r)]['Absolute Error'] for r in test_r]
    y = [np.sqrt(((ele ** 2).sum()) / len(ele)) for ele in abs_error_data]
    plt.plot(x, y, label=model, marker='o')  
plt.xlabel('Risk-Free interest rate')
plt.ylabel('RMSE')
plt.title('Model performance across different Risk-Free interest rates')
plt.legend()  
plt.grid(True) 
plt.show()