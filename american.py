import numpy as np
from scipy.stats import norm
import xgboost as xgb
from sklearn.neural_network import MLPRegressor

# Uses machine learning for regression and optimal stopping
def longstaff_schwartz (S0, r, sigma, T, K, n_trials, n_timesteps, option_type, ml_model):
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
    if option_type == 'put':
        payoffs[:,n_timesteps-1] = np.maximum(K - ST, 0)

    for t in range(n_timesteps-2,-1,-1):
        itm_indices = np.where(S[:,t] > 0)
        otm_indices = np.where(S[:,t] <= 0)
        X = S[itm_indices, t].reshape(-1, 1)
        y = payoffs[itm_indices,t+1] * np.exp(-r*dt)
        if ml_model == "xgboost":
            model = xgb.XGBRegressor()
        if ml_model == "mlp":
            model = MLPRegressor(
                hidden_layer_sizes=(64, 64),  
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        model.fit(X, y)
        future_val = model.predict(X).flatten()
        if option_type == 'call':
            current_val = np.maximum(S[itm_indices, t] - K, 0).flatten()
        if option_type == 'put':
            current_val = np.maximum(K - S[itm_indices, t], 0).flatten()

        itm = itm_indices[0]  
        should_exercise = current_val > future_val

        payoffs[itm[should_exercise], t] = current_val[should_exercise]
        payoffs[itm[~should_exercise], t] = payoffs[itm[~should_exercise], t+1] * np.exp(-r*dt)

        payoffs[otm_indices, t] = payoffs[otm_indices, t+1]


    profit = np.mean(payoffs[:,0])

    SE = np.std(payoffs[:,0], ddof=1) / np.sqrt(n_trials)
    lower = profit-1.96*SE
    upper = profit+1.96*SE

    return profit, lower, upper