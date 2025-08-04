import numpy as np
from scipy.stats import norm

# Uses Geometric Brownian Motion for Monte Carlo Simulation 
def monte_carlo(S0, r, sigma, T, K, N, option_type):
    Z = np.random.normal(0, 1, N)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    if option_type == 'put':
        payoffs = np.maximum(K - ST, 0)

    profit = np.mean(payoffs) * np.exp(-r*T)

    SE = np.std(payoffs, ddof=1) / np.sqrt(N)
    lower = profit-1.96*SE
    upper = profit+1.96*SE

    return profit, lower, upper

# Uses closed-form formula of Black-Scholes for european options 
def black_scholes(S0, r, sigma, T, K, option_type):
    d1 = (np.log(S0/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        C = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return C
    if option_type == 'put':
        P = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
        return P
