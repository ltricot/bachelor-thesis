import pandas as pd
import ujson as json

import os


DATADIR  = os.path.join(os.path.dirname(__file__), 'data/raw/')
CACHEDIR = '/tmp/arch/'

if not os.path.exists(CACHEDIR):
    os.makedirs(CACHEDIR)


# btcusdt -> file name
REGISTRY = {
    os.path.basename(fn).split('-')[0]: os.path.join(DATADIR, fn)
    for fn in os.listdir(DATADIR)
}


def read_to_table(src, cachedir=CACHEDIR):

    # cache in hdfs: much faster to read
    cache = os.path.join(cachedir, f'{os.path.basename(src)}.h5')
    if os.path.exists(cache):
        try:
            with pd.HDFStore(cache) as st:
                table = st.select('_', columns=['open_time', 'open'])
                return table
        except:
            os.remove(cache)

    # specify numerics where the json object stores strings
    columns = {
        'open_time': int, 'open': float, 'high': float, 'low': float,
        'close': float, 'volume': float, 'close_time': int,
        'quote_asset_volume': float, 'n_trades': int,
        'taker_buy_base_asset_volume': float,
        'taker_buy_quote_asset_volume': float, 'ignore': str,
    }

    # read and store in cache
    with open(src) as f, pd.HDFStore(cache) as st:
        df = pd.DataFrame(
            data=[json.loads(line) for line in f],
            columns=tuple(columns.keys()),
        )

        df = df.astype(columns)
        st.put('_', df, format='t', data_columns=True)

    return df


from pandas_datareader import DataReader
import datetime


def indexwith(pairs, name):
        dfs = []
        for p in pairs:
            dfs.append(priceof(p)[['open']])

        df = dfs.pop().pct_change()
        for n in dfs:
            df.add(n.pct_change(), fill_value=0)

        df = df.dropna() / len(pairs)
        df.open = (1 + df.open).cumprod()
        df.name = name
        return df


# convenience
def priceof(pair):
    # mark my words, the creation of these indices will go down in history
    if pair == '^TCFA':  # Tricot Cryptographic Finance Average
        # return indexwith([
        #     'btcusdt',  # bitcoin and all that follows bitcoin
        #     # 'crousdt',  # crypto.com coin, n°1 in basic crypto services
        #     'bnbusdt',  # health of largest crypto exchange in the world
        #     'zrxusdt',  # decentralized exchange protocol token
        #     'xrpusdt',  # a different approach to a cryptocurrency
        # ], '^TCFA')
        return indexwith([
            pair for pair in REGISTRY
            if pair.endswith('usdt')
        ], '^TCFA')

    if pair == '^TCDPIA':  # Tricot Cryptographic Derivative Projects Internal Average
        return indexwith([
            'repbtc',  # augur's reputation token
            # 'mkrbtc',  # maker dao token
            'batbtc',  # basic attention token
            'enjbtc',  # gaming community thing
            'omgbtc',  # OmiseGo
        ], '^TCDPIA')

    # not in binance ? yahoo finance
    if pair not in REGISTRY:
        start = datetime.datetime(2010, 1, 1)
        end = datetime.datetime.today()

        df = DataReader(pair, 'yahoo', start, end)
        df = df.rename(columns={'Open': 'open'})
        df.name = pair

        return df

    # binance pairs
    fn = REGISTRY[pair]
    df = read_to_table(fn)

    df.open_time = pd.to_datetime(df.open_time, unit='ms')
    df.set_index('open_time', inplace=True)
    df.sort_index(inplace=True)

    df.name = pair  # does not persist
    return df
