import pytest
import numpy as np
from src.unit1_2_random_variables.distributions import BinomialDistribution, NormalDistribution, UniformDistribution


def test_binomial_pmf():
    """Verify PMF(k=2, n=5, p=0.5) ≈ 0.3125"""
    dist = BinomialDistribution(n=5, p=0.5)
    result = dist.pmf(2)
    assert abs(result - 0.3125) < 1e-6, f"Expected ~0.3125, got {result}"


def test_binomial_pmf_boundary():
    """PMF at k=0 and k=n"""
    dist = BinomialDistribution(n=3, p=0.5)
    assert abs(dist.pmf(0) - 0.125) < 1e-6
    assert abs(dist.pmf(3) - 0.125) < 1e-6


def test_binomial_cdf():
    """CDF should be monotonically increasing and CDF(n) = 1"""
    dist = BinomialDistribution(n=5, p=0.3)
    prev = 0.0
    for k in range(6):
        cdf_k = dist.cdf(k)
        assert cdf_k >= prev - 1e-9
        prev = cdf_k
    assert abs(dist.cdf(5) - 1.0) < 1e-6


def test_binomial_mean_variance():
    """mean = n*p, variance = n*p*(1-p)"""
    dist = BinomialDistribution(n=10, p=0.4)
    assert abs(dist.mean() - 4.0) < 1e-9
    assert abs(dist.variance() - 2.4) < 1e-9


def test_binomial_sample():
    """Sample shape and range check"""
    dist = BinomialDistribution(n=10, p=0.5)
    samples = dist.sample(100)
    assert len(samples) == 100
    assert all(0 <= s <= 10 for s in samples)


def test_normal_pdf():
    """Standard normal PDF at x=0 ≈ 0.3989"""
    dist = NormalDistribution(mu=0, sigma=1)
    result = dist.pdf(0)
    expected = 1.0 / np.sqrt(2 * np.pi)
    assert abs(result - expected) < 1e-6, f"Expected ~{expected}, got {result}"


def test_normal_cdf():
    """Standard normal CDF at x=0 ≈ 0.5"""
    dist = NormalDistribution(mu=0, sigma=1)
    result = dist.cdf(0)
    assert abs(result - 0.5) < 1e-4, f"Expected ~0.5, got {result}"


def test_normal_cdf_symmetry():
    """CDF(-x) = 1 - CDF(x)"""
    dist = NormalDistribution(mu=0, sigma=1)
    for x in [0.5, 1.0, 1.96, 2.5]:
        assert abs(dist.cdf(-x) - (1 - dist.cdf(x))) < 1e-4


def test_normal_cdf_known_values():
    """CDF(1.96) ≈ 0.975"""
    dist = NormalDistribution(mu=0, sigma=1)
    assert abs(dist.cdf(1.96) - 0.975) < 0.01


def test_normal_sample():
    """Sample mean and std check"""
    np.random.seed(42)
    dist = NormalDistribution(mu=5, sigma=2)
    samples = dist.sample(10000)
    assert len(samples) == 10000
    assert abs(np.mean(samples) - 5) < 0.1
    assert abs(np.std(samples) - 2) < 0.1


def test_uniform_pdf():
    """Uniform(0,1) PDF at any x in [0,1] = 1.0"""
    dist = UniformDistribution(a=0, b=1)
    for x in [0.1, 0.5, 0.9]:
        assert abs(dist.pdf(x) - 1.0) < 1e-9
    assert dist.pdf(-0.1) == 0.0
    assert dist.pdf(1.1) == 0.0


def test_uniform_cdf():
    """Uniform CDF: CDF(x) = (x-a)/(b-a) for x in [a,b]"""
    dist = UniformDistribution(a=0, b=4)
    assert abs(dist.cdf(0) - 0.0) < 1e-9
    assert abs(dist.cdf(2) - 0.5) < 1e-9
    assert abs(dist.cdf(4) - 1.0) < 1e-9


def test_uniform_mean_variance():
    dist = UniformDistribution(a=2, b=8)
    assert abs(dist.mean() - 5.0) < 1e-9
    assert abs(dist.variance() - 3.0) < 1e-9
