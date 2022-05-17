from curses import panel
from matplotlib import projections
from scipy.stats import multivariate_normal
import numpy as np
import cvxpy as cp
from IPython import embed as e


def nonsmooth_exp(beta, q, yk, baseline, m, R, panelty=3e3):
    grad = m**10 / np.linalg.norm(m**10,1)

    def f_beta(_input):
        x, alpha = _input[:3], _input[3]

        # Absolute value
        x = np.abs(x)

        # Normalize 
        # x = x/np.linalg.norm(x, 1)

        temp_sum = 0
        for k in range(q):
            temp_sum += np.maximum(-x.T @ yk[k] - alpha, 0)
        
        s_ = alpha + 1/(q * (1-beta)) * temp_sum 

        if - np.sum(x * m) + R > 0:
            
            s_ += panelty * (- np.sum(x * m) + R)
        return s_
    def comp_gradient(_input):
        x, alpha = _input[:-1], _input[-1]
        mask = np.sign(x)
        x = np.abs(x)

        norm_ratio = (np.sum(x) - x)/(np.sum(x) ** 2)


        assert (x>=0).all()

        gradient = np.zeros(4)
        gradient[-1] = 1

        temp_sum = np.zeros(4)
        for k in range(q):
            
            g = np.concatenate([-yk[k], [-1]])
            # e() or b
            if x.T @ -yk[k] - alpha > 0:
                temp_sum += g
            elif x.T @ -yk[k] - alpha == 0:
                temp_sum += np.random.rand() * g


        gradient += 1/(q*(1-beta)) * temp_sum


        # gradient[:-1] *= proj_ratio

        # gradient[:-1] *= norm_ratio
        if - np.sum(x * m) + R > 0:
            # e() or b
            # print("proj", - np.sum(x * m) + R - 0.01 )
            gradient[:-1] += panelty *  (-m ** 2)

        gradient[:-1] *= mask

        return gradient

    x_s = np.array([0.2, 0.4, 0.4, 0])
    g_s = comp_gradient(x_s)
    H_s = np.eye(4)
    rho = 1e-1
    lam = 1e-2
    tol = 1e-3
    eps = 1e-4
    done = False

    inner_max = 100



    last_f = np.inf
    last_x = None
    for s in range(10000):
        # print(s)

        x_i = np.zeros(4)
        g_i = np.zeros(4)
        H_i = H_s
        for s2 in range(inner_max):
            # e() or b
            # print("G_S", g_s)

            x_i = x_s - rho * H_i @ g_s
            x_i[:-1] /= np.linalg.norm(x_i[:-1], 1) # Normalize



            g_i = comp_gradient(x_i)
            # print("g_i", g_i)
            

            if np.linalg.norm(g_i, 2) < tol:
                done = True
                break

            xi_i = g_i / np.linalg.norm(g_i, 2)
            H_i += + lam * np.outer(xi_i, g_s)

            if f_beta(x_i) < f_beta(x_s) - eps:
                break

            # if s2 == inner_max - 1:
            #     print("not good direction")

        x_s = x_i
        # x_s[:-1] = np.abs(x_s[:-1]) / np.linalg.norm(x_s[:-1], 1)
        # e() or b
        g_s = g_i
        H_s = H_i

        # print("F_beta", f_beta(x_s))
        # print(x_s)
        # print("Residual:", np.sum(x_s[:-1] * m) - R)
        # print(((np.sum(x_s[:-1] * m) - R < 1e-5) ) )
        if ((last_f - f_beta(x_s)) < 1e-5 and np.sum(last_x[:-1] * m) >= R) :
            # print(f"New {f_beta(x_s)} Old {last_f}")
            # print("Early stop")
            break
        else:
            last_f = f_beta(x_s)
            last_x = x_s

        # if done: break
    str_x = ' '.join(map(lambda i:format(i, '.5f'),last_x[:-1]))
    alpha = last_x[-1]
    CVaR = f_beta(last_x)

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
            nonsmooth_exp(beta, q, yk, baselines[i])
            break
        # break