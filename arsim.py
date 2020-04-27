import numpy as np
import matplotlib.pyplot as plt


def AR(params, n, sig=1, errf=np.random.normal):
    seq = [0] * len(params)
    errs = errf(size=n, scale=np.sqrt(sig))
    for i in range(n):
        seq.append(np.dot(seq[-len(params):], params[::-1]) + errs[i])
    return np.asarray(seq[-n:]), errs

def MA(params, n, errf=np.random.normal):
    seq = []
    errs = errf(size=n + len(params))
    for i in range(n):
        seq.append(np.dot(errs[i:i+len(params)], params[::-1]))
    return np.asarray(seq), errs[-n:]

def plotvc():
    for seq in [
        AR([0.5], n=500)[0],
        MA(np.cumprod(0.99 * np.ones(30)), n=500)[0],
    ]:
        seq = seq**2

        from stylized import autocorr

        fig, ax = plt.subplots()
        fig.set_tight_layout(True)

        ml = 150
        lags = np.arange(1, ml + 1)
        ax.bar(
            lags,
            autocorr(seq, ml),
            color='black',
            width=0.3,
        )

        plt.show()

def plottails():
    from scipy.stats import kurtosis, gaussian_kde, norm

    for func in (AR, MA):
        for errf in (np.random.normal, np.random.laplace):
            seq, errs = func([0.5], n=5000, errf=errf)

            fig, ax = plt.subplots()
            fig.set_tight_layout(True)

            seq = np.sort(seq)
            seq = (seq - np.mean(seq)) / np.std(seq)
            kernel = gaussian_kde(seq)

            curve = kernel(seq)
            ax.plot(seq, curve, label='normalized returns')

            xx = np.linspace(*ax.get_xlim(), 300)
            ax.plot(
                xx, norm.pdf(xx),
                label='gaussian', linestyle='--', color='black'
            )

            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticks([])

            ax.set_title((
                f'Distribution of returns of '
                f'${func.__name__}$ process with {errf.__name__} error '
                f'distribution'
            ))
            ax.legend()

            title = f'plots/{func.__name__}-{errf.__name__}.png'
            print(title, kurtosis(seq))
            fig.savefig(title)

def plotsim():
    for i, (params, func) in enumerate([
            (np.asarray([1, -1/4]), AR),
            (np.asarray([0.5]), AR),
            (np.asarray([1, 2]), MA),
            (np.asarray([1, 1, 1, 1]), MA),
    ]):
        seq, *_ = func(params, 200)

        fig, ax = plt.subplots()
        ax.plot(seq, color='black')

        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.set_title((
            f'Simulation of {func.__name__} '
            f'process with parameters {", ".join(map(str, params))}'
        ))

        fig.savefig(f'plots/{func.__name__}{i}.png')


if __name__ == '__main__':
    plotvc()
