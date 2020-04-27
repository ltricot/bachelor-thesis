import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import numpy as np
import scipy.stats

import pandas as pd


def fitar(seq, p=1):
    A = np.vstack([seq[i:-p + i] for i in range(p)]).T
    y = seq[p:]
    return lstsq(A, y, rcond=None)[0], A, y

def fitarma(seq, p=1, q=1):
    A = np.vstack([
        seq[i:i+p]
        for i in range(len(seq) - p)
    ])
    y = seq[p:]
    x = lstsq(A, y, rcond=None)[0]

    res = y - A @ x
    seq = y

    A_ar = np.vstack([
        seq[i:i+p]
        for i in range(max(p, q) - p, len(seq) - p)
    ])

    A_ma = np.vstack([
        res[i:i+p]
        for i in range(max(p, q) - q, len(seq) - q)
    ])

    A = np.hstack((A_ar, A_ma))
    y = seq[max(p, q):]

    x = lstsq(A, y, rcond=None)[0]
    return x, A, y

def sim_garch(As, Bs, n):
    k = max(len(As), len(Bs))
    eps = np.random.normal(size=n + 300 + k)
    ep2 = eps**2
    h = np.ones_like(eps)

    for i in range(k, n + 300 + k):
        h[i] = np.sqrt(
            Bs[0] +
            np.dot(Bs[1:], ep2[i - len(Bs) + 1:i][::-1]) +
            np.dot(As, h[i - len(As):i][::-1])
        )
        eps[i] *= h[i]
        ep2[i] = eps[i]**2

    return eps[-n:]

def mom2_garch(seq, p=1, q=1, c=3.7, g=True):
    seq = np.square(seq)

    k = 10 * (p + q)

    A = np.vstack([
        [1, *seq[i:i+k]]
        for i in range(len(seq) - k)
    ])
    y = seq[k:]

    ix = y < c
    Ap, yp = A[ix], y[ix]

    x = lstsq(Ap, yp, rcond=None)[0]

    if not g:
        return x, A, y

    sigma = A @ x
    seq = y

    A_ar = np.vstack([
        sigma[i:i+p]
        for i in range(max(p, q) - p, len(seq) - p)
    ])

    A_ma = np.vstack([
        [1, *seq[i:i+q]]
        for i in range(max(p, q) - q, len(seq) - q)
    ])

    A = np.hstack((A_ar, A_ma))
    y = y[max(p, q):]

    # ix = y < c
    # Ap, yp = A[ix], y[ix]
    Ap, yp = A, y

    x = lstsq(Ap, yp, rcond=None)[0]
    return x, A, y


if __name__ == '__main__':
    from arch import arch_model
    from contextlib import redirect_stdout
    from io import StringIO

    As = np.asarray([0.1, 0.1, 0.1])
    Bs = np.asarray([0.1, 0.1])
    n = 1000

    b01 = []
    b02 = []

    for _ in range(300):
        eps = sim_garch(As, Bs, n)
        par1, *_ = mom2_garch(eps, p=1, q=1)
        b01.append(par1[0])

        # just shut up
        with redirect_stdout(StringIO()):
            g = arch_model(eps, p=1, q=1)
            par2 = g.fit(disp='off').params
            par2 = par2[[3, 1, 2]]
        b02.append(par2[0])

    plt.plot(eps)
    plt.show()

    print(np.mean(b01))
    plt.hist(b01, bins=50)
    plt.show()

    print(np.mean(b02))
    plt.hist(b02, bins=50)
    plt.show()

def _():
    from read import priceof
    from arsim import AR


    past = 200
    pos = np.random.randint(400)

    btcc = priceof('btcusdt').open.iloc[::60*24]
    btc = btcc.pct_change().to_numpy()[-past-pos:-pos]
    btc_p = btc #btcc.pct_change().to_numpy()[-pos:-pos+past]

    p, q = 10, 10
    pp, qq = 10, 10
    par, A, y = fitarma(btc, p=p, q=q)
    # _, Ap, yp = fitarma(btc_p, p=p, q=q)
    Ap, yp = A, y
    pred = A @ par
    predp = Ap @ par
    res = y - pred
    resp = yp - predp

    pgarch, B, u = mom2_garch(res, p=pp, q=qq)
    # _, Bp, up = mom2_garch(resp, p=pp, q=qq)
    Bp, up = B, u
    vol = np.sqrt(B @ pgarch)
    volp = np.sqrt(Bp @ pgarch)

    plt.plot(np.arange(len(btc_p)), btc_p, label='btc')
    plt.plot(np.arange(p + max(p, q), len(btc_p)), predp, label='predictions')
    plt.plot(np.arange(p + max(p, q) + pp + qq, len(btc_p)), volp, label='volatility')

    rvol = pd.Series(resp).rolling(20).std()
    plt.plot(np.arange(len(btc_p) - len(rvol), len(btc_p)), rvol, label='rolling standard deviation')
    plt.legend()

    plt.show()
