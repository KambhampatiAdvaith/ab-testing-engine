import numpy as np


def _log_factorial(n: int) -> float:
    """Compute log(n!) using sum of logs for numerical stability."""
    if n <= 0:
        return 0.0
    return float(np.sum(np.log(np.arange(1, n + 1))))


def _log_comb(n: int, k: int) -> float:
    """Compute log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)"""
    return _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)


def _erf(x: float) -> float:
    """
    Abramowitz and Stegun approximation of the error function.
    Max error: 1.5e-7
    """
    if x == 0:
        return 0.0
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    result = 1.0 - poly * np.exp(-x * x)
    return float(np.sign(x)) * result


class BinomialDistribution:
    """
    Binomial distribution: models number of successes in n independent Bernoulli trials.

    PMF: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
    Mean: n*p
    Variance: n*p*(1-p)
    """

    def __init__(self, n: int, p: float):
        """
        Initialize the Binomial distribution.

        Args:
            n: Number of trials (non-negative integer)
            p: Probability of success per trial (0 <= p <= 1)
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self.n = n
        self.p = p

    def pmf(self, k: int) -> float:
        """
        Probability mass function P(X=k).

        Uses log-factorial for numerical stability:
        log P(X=k) = log C(n,k) + k*log(p) + (n-k)*log(1-p)

        Args:
            k: Number of successes (0 <= k <= n)

        Returns:
            Probability P(X=k)
        """
        if k < 0 or k > self.n:
            return 0.0
        # Handle edge cases where p=0 or p=1
        if self.p == 0:
            return 1.0 if k == 0 else 0.0
        if self.p == 1:
            return 1.0 if k == self.n else 0.0
        log_pmf = _log_comb(self.n, k) + k * np.log(self.p) + (self.n - k) * np.log(1 - self.p)
        return float(np.exp(log_pmf))

    def cdf(self, k: int) -> float:
        """
        Cumulative distribution function P(X <= k).

        Args:
            k: Upper bound for the sum

        Returns:
            P(X <= k)
        """
        if k < 0:
            return 0.0
        if k >= self.n:
            return 1.0
        return float(sum(self.pmf(i) for i in range(int(k) + 1)))

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples using inverse transform sampling.

        For each uniform U, find smallest k where CDF(k) >= U.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of sampled values
        """
        u = np.random.uniform(0, 1, n_samples)
        samples = np.zeros(n_samples, dtype=int)
        for i, ui in enumerate(u):
            cumprob = 0.0
            for k in range(self.n + 1):
                cumprob += self.pmf(k)
                if cumprob >= ui:
                    samples[i] = k
                    break
        return samples

    def mean(self) -> float:
        """Mean = n*p"""
        return float(self.n * self.p)

    def variance(self) -> float:
        """Variance = n*p*(1-p)"""
        return float(self.n * self.p * (1 - self.p))


class NormalDistribution:
    """
    Normal (Gaussian) distribution: N(mu, sigma^2).

    PDF: (1 / (sigma * sqrt(2*pi))) * exp(-(x-mu)^2 / (2*sigma^2))
    CDF: 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    """

    def __init__(self, mu: float, sigma: float):
        """
        Initialize Normal distribution.

        Args:
            mu: Mean
            sigma: Standard deviation (must be > 0)
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x: float) -> float:
        """
        Probability density function at x.

        Args:
            x: Point at which to evaluate the PDF

        Returns:
            PDF value at x
        """
        z = (x - self.mu) / self.sigma
        return float((1.0 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z ** 2))

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function at x.

        Uses error function approximation (Abramowitz & Stegun):
        CDF(x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))

        Args:
            x: Point at which to evaluate the CDF

        Returns:
            CDF value at x
        """
        z = (x - self.mu) / (self.sigma * np.sqrt(2))
        return float(0.5 * (1.0 + _erf(z)))

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples using Box-Muller transform.

        Z1 = sqrt(-2*ln(U1)) * cos(2*pi*U2)
        Z2 = sqrt(-2*ln(U1)) * sin(2*pi*U2)

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of sampled values
        """
        n = n_samples + (n_samples % 2)  # ensure even
        U1 = np.random.uniform(0, 1, n // 2)
        U2 = np.random.uniform(0, 1, n // 2)
        Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
        Z2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
        Z = np.concatenate([Z1, Z2])[:n_samples]
        return self.mu + self.sigma * Z

    def mean(self) -> float:
        """Mean = mu"""
        return self.mu

    def variance(self) -> float:
        """Variance = sigma^2"""
        return self.sigma ** 2


class UniformDistribution:
    """
    Continuous Uniform distribution on [a, b].

    PDF: 1/(b-a) for x in [a, b], 0 otherwise
    CDF: 0 for x<a, (x-a)/(b-a) for x in [a,b], 1 for x>b
    """

    def __init__(self, a: float, b: float):
        """
        Initialize Uniform distribution.

        Args:
            a: Lower bound
            b: Upper bound (must be > a)
        """
        if b <= a:
            raise ValueError(f"b must be greater than a, got a={a}, b={b}")
        self.a = a
        self.b = b

    def pdf(self, x: float) -> float:
        """
        Probability density function at x.

        Args:
            x: Point at which to evaluate the PDF

        Returns:
            PDF value at x
        """
        if self.a <= x <= self.b:
            return 1.0 / (self.b - self.a)
        return 0.0

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function at x.

        Args:
            x: Point at which to evaluate the CDF

        Returns:
            CDF value at x
        """
        if x < self.a:
            return 0.0
        if x > self.b:
            return 1.0
        return float((x - self.a) / (self.b - self.a))

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples using numpy.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of sampled values
        """
        return np.random.uniform(self.a, self.b, n_samples)

    def mean(self) -> float:
        """Mean = (a + b) / 2"""
        return (self.a + self.b) / 2

    def variance(self) -> float:
        """Variance = (b - a)^2 / 12"""
        return (self.b - self.a) ** 2 / 12
