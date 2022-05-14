from scipy.stats import multivariate_normal
import numpy as np
import cvxpy as cp

m = np.array([0.0101110, 0.0043532, 0.0137058])
V = np.array([[0.00324625, 0.00022983, 0.00420395], 
              [0.00022983, 0.00049937, 0.00019247],
              [0.00420395, 0.00019247, 0.00764097]])

qs=[1000, 3000, 5000, 10000, 20000]
betas = [0.9, 0.95, 0.99]
R = 0.011

y_dist = multivariate_normal(mean=m, cov=V)

baselines = [(0.067847, 0.096975), 
             (0.090200, 0.115908), 
             (0.132128, 0.152977)]

def experiment(beta, q, yk, baseline):
    uk = cp.Variable(q)
    x = cp.Variable(3)
    alpha = cp.Variable()

    prob = cp.Problem(
        cp.Minimize(alpha + 1/(q * (1-beta)) * cp.sum(uk)),
        [uk >= 0] + 
        [x.T @ yk[i] + alpha + uk[i] >=0 for i in range(q)] +
        [-x.T @ m <= -R] +
        [cp.sum(x) == 1, x[0] >= 0, x[1] >= 0, x[2] >= 0]
        # x.T @ yk[0] + alpha 

    )
    assert (prob.is_dcp())

    prob.solve()

    str_x = ' '.join(map(lambda i:format(i, '.5f'),x.value))

    (base_VaR, base_CVaR) = baseline

    print(f"{beta}\t{q}\t{str_x}\t \
        {alpha.value:.5f}\t{(alpha.value - base_VaR)/base_VaR*100:.2f}\t \
        {prob.value:.5f}\t{(prob.value - base_CVaR)/base_CVaR*100:.2f}")


print("beta\tSample#\tS&P\tBond\tSmall Cap\tValue-at-Risk\tVaR Dif(%)\tCond VaR\tCVaR Dif(%)")

for i, beta in enumerate(betas):
    for q in qs:
        yk = y_dist.rvs(q)
        experiment(beta, q, yk, baselines[i])
    #     break
    # break