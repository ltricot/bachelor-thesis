import numpy as np
from read import priceof


def mat(seq, p):
    shifts = [seq[p-i-1:-i-1] for i in range(p)]
    return np.vstack((*shifts, np.ones(len(seq) - p))).T

def ar(seq, p=1):
    return np.linalg.lstsq(mat(seq, p), seq[p:], rcond=None)[0]


returns = priceof('btcusdt').open.iloc[::60*24].pct_change()
params = ar(returns.to_numpy()[-300:], p=2)
residuals = returns[2:] - mat(returns, 2) @ params[:, np.newaxis]

plt.plot(residuals)
plt.show()
