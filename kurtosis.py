from read import priceof
import numpy as np


ccs = (
    '^TCFA', '^FCHI', '^DJI',
    'btcusdt', 'xrpusdt', 'ethusdt',
)

toks = (
    '^TCDPIA', 'repbtc', 'batbtc', 'enjbtc', 'omgbtc',
)

for pair in toks:
    asset = priceof(pair)

    scales = (1,)
    if pair not in ('^FCHI', '^DJI'):
        scales = np.cumprod((1, 60, 24))

    for scale in scales:
        returns = asset.open.iloc[::scale].pct_change()
        kurt = returns.kurtosis()

        print(f'{pair} fisher kurtosis, returns interval {scale}: {kurt}')
