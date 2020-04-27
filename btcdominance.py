from datetime import datetime
from read import REGISTRY, priceof
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def assets(pairs):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    for pair in pairs:
        df = priceof(pair).iloc[::60].iloc[-10000:]
        df.open = df.open / df.open[0]  # normalize to be comparable
        dates = df.index.strftime('%Y-%m-%d')

        ax.plot(
            dates, df.open,
            # marker='.', markersize=1,
            ls='-', lw=1,
            label=pair,
        )

    ax.set_title('Normalized evolution of assets compared to bitcoin')

    ax.tick_params(axis='x', labelrotation=40)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    return fig


def corrmap():
    curves = [key for key in REGISTRY if key.endswith('usdt')]

    dfs = []
    for curve in curves:
        df = priceof(curve)[['open']].rename(columns={'open': curve})
        print(curve, df.index[0])
        if df.index[0] < datetime(2019, 6, 1):
            dfs.append(df)

    df = dfs.pop().join(dfs)
    corr = df.corr()
    sns.heatmap(
        corr,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
    )

    plt.show()


fig = assets(['btcusdt', 'ethusdt', 'ltcusdt', 'bnbusdt', 'xmrusdt'])
fig.savefig('plots/btcdominance.png')
plt.close(fig)

fig = assets(['gntbtc', 'fttbtc', 'chzbtc'])
fig.savefig('plots/btcindependent.png')
