import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from read import priceof, REGISTRY


INTERVALS = {
    'minute': 1, 'hourly': 60, 'daily': 60 * 24, 'weekly': 60 * 24 * 7
}


def returns(df, interval='daily'):
    name = df.name
    df = df.iloc[::INTERVALS[interval]].tail(500)

    dates = df.index.strftime('%Y-%m-%d %r')
    returns = df.open.pct_change()

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_title(f'{name} returns at {interval} interval')

    ax.plot(dates, returns, linewidth=1, c='grey')

    ax.tick_params(axis='x', labelrotation=30)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig

# TODO: rewrite
def autocorr(arr, max_lag):
    s = pd.Series(arr)
    ac = []
    for lag in range(1, max_lag + 1):
        ac.append(s.autocorr(lag=lag))
    return np.asarray(ac)

# TODO: rewrite
def corr(x, y, max_lag):
    x, y = pd.Series(x), pd.Series(y)
    c = []
    for lag in range(1, max_lag + 1):
        c.append(x.iloc[:-lag].corr(y.iloc[lag:]))
    return np.asarray(c)

def autocorrelation_squared(df, interval, max_lag=60):
    lags = np.arange(1, max_lag + 1)
    returns = df.open.iloc[::INTERVALS[interval]].pct_change()**2

    # must have more returns that lag
    if len(returns) < (max_lag + 1) * 4:
        raise ValueError

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_title(
        f'autocorrelation of squared {df.name} returns at {interval} scale')

    ax.bar(
        lags,
        autocorr(returns, max_lag),
        color='black',
        width=0.1,
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig

def autocorrelation(df, interval, max_lag=60):
    lags = np.arange(1, max_lag + 1)
    returns = df.open.iloc[::INTERVALS[interval]].pct_change()

    # must have more returns that lag
    if len(returns) < (max_lag + 1) * 4:
        raise ValueError

    randoms = np.random.normal(size=len(returns), scale=returns.std())

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_title(
        f'autocorrelation of {df.name} compared to gaussian\n'
        f'at {interval} scale'
    )

    width = 0.1
    ax.bar(
        lags + width / 2,
        autocorr(randoms, max_lag),
        width=width,
        color='black',
        label='gaussian'
    )

    ax.bar(
        lags - width / 2,
        autocorr(returns, max_lag),
        width=width,
        color='orange',
        label='returns'
    )

    c = 1.96 / np.sqrt(len(returns))
    ax.axhline(c, linestyle='--', linewidth=1, color='black')
    ax.axhline(-c, linestyle='--', linewidth=1, color='black')

    ax.set_ylim([-3 * c, 3 * c])
    ax.legend()

    return fig

def leverage(df, interval, max_lag=60):
    lags = np.arange(1, max_lag + 1)
    returns = df.open.iloc[::INTERVALS[interval]].pct_change()

    plus = returns.clip(lower=0)
    minus = -returns.clip(upper=0)

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.set_title(f'leverage effect for {df.name} at {interval} scale')

    width = 0.1
    ax.bar(
        lags - width / 2,
        corr(returns, returns.abs(), max_lag=max_lag),
        width=width,
        color='black',
        label='$r_+$',
    )

    # ax.bar(
    #     lags + width / 2,
    #     corr(minus, returns.abs(), max_lag=max_lag),
    #     width=width,
    #     color='orange',
    #     label='$r_-$',
    # )

    ax.legend()

    return fig

def fat_tailed(df, interval):
    from scipy.stats import gaussian_kde, norm

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    for interval, step in INTERVALS.items():

        # too peaked
        if interval == 'minute':
            continue

        # make returns comparable to standard normal
        rets = df.open.iloc[::step].pct_change()
        rets = np.sort(rets.to_numpy()[1:])  # remove nan
        rets = (rets - np.mean(rets)) / np.std(rets)
        kernel = gaussian_kde(rets)

        # just so this doesn't take 8 hours
        if len(rets) > 300:
            shown = rets[::len(rets) // 300]
        else:
            shown = rets

        curve = kernel(shown)
        ax.plot(shown, curve, label=interval)

    ax.set_xlim(-5, 5)

    xx = np.linspace(*ax.get_xlim(), 300)
    ax.plot(xx, norm.pdf(xx), label='gaussian', linestyle='--', color='black')

    ax.legend()

    # pretty axes
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.get_yaxis().set_ticks([])

    return fig


if __name__ == '__main__':
    import os

    PLOTDIR = 'plots'
    plots = [
        returns, autocorrelation, autocorrelation_squared,
        leverage, fat_tailed
    ]


    for pair in [
            'btcusdt', 'xrpusdt', '^TCFA', '^TCDPIA', '^DJI', '^FCHI',
            'bnbusdt', 'zrxusdt',
        ]:

        df = priceof(pair)
        print(df.columns)

        for plot in plots:
            for interval in ('minute', 'hourly', 'daily'):
                try:
                    fig = plot(df, interval=interval)
                except ValueError:
                    continue

                fig.savefig(os.path.join(
                    PLOTDIR,
                    f'{pair}-{plot.__name__}-{interval}.png'
                ))

                plt.close(fig)

                if plot is fat_tailed:
                    break
