from scipy.optimize import minimize
from matplotlib import projections
from scipy.stats import multivariate_normal
import numpy as np
import cvxpy as cp
from IPython import embed as e

def bfgs_exp(beta, q, yk, baseline, m, R, panelty=3e3):
    def f_beta(_input):
        x, alpha = _input[:3], _input[3]

        # Absolute value
        x = np.abs(x)

        # Normalize 
        x = x/np.linalg.norm(x, 1)

        temp_sum = 0
        for k in range(q):
            temp_sum += np.maximum(-x.T @ yk[k] - alpha, 0)
        
        s_ = alpha + 1/(q * (1-beta)) * temp_sum 

        if - np.sum(x * m) + R > 0:
            
            s_ += panelty * (- np.sum(x * m) + R)
        return s_
    
    x_0 = np.array([0.2, 0.4, 0.4, 0])
    ret = minimize(f_beta, x_0, method='BFGS')
    x = ret.x
    x = np.abs(x)

    # Normalize 
    x[:-1] = x[:-1]/np.linalg.norm(x[:-1], 1)
    # e() or b

    str_x = ' '.join(map(lambda i:format(i, '.5f'),x[:-1]))
    alpha = x[-1]
    CVaR = ret.fun
    # e() or b

    (base_VaR, base_CVaR) = baseline
    print(f"{beta}\t{q}\t{str_x}\t{alpha:.5f}\t \
    {(alpha - base_VaR)/base_VaR*100:.2f}\t \
    {CVaR:.5f}\t{(CVaR - base_CVaR)/base_CVaR*100:.2f}")


if __name__ == "__main__":

    m = np.array([0.0101110, 0.0043532, 0.0137058])
    V = np.array([[0.00324625, 0.00022983, 0.00420395], 
                [0.00022983, 0.00049937, 0.00019247],
                [0.00420395, 0.00019247, 0.00764097]])

    qs=[1000, 3000, 5000, 10000, 20000]
    betas = [0.9, 0.95, 0.99]
    R = 0.011

    panelty = 3e3

    y_dist = multivariate_normal(mean=m, cov=V)

    baselines = [(0.067847, 0.096975), 
                (0.090200, 0.115908), 
                (0.132128, 0.152977)]

    print("beta\tSample#\tS&P\tBond\tSmall\tValue-at-Risk\tVaR Dif(%)\tCond VaR\tCVaR Dif(%)")

    for i, beta in enumerate(betas):
        for q in qs:
            np.random.seed(2021)
            yk = y_dist.rvs(q)
            bfgs_exp(beta, q, yk, baselines[i], m, R)
            break
        break