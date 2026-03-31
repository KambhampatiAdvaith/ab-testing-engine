import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from math import lgamma


class BayesianABTest:
    """
    Bayesian A/B testing using Beta-Binomial conjugate model.

    Prior: Beta(alpha, beta)  -- default: uniform Beta(1,1)
    Posterior after observing s successes and f failures:
        Beta(alpha + s, beta + f)

    This class supports:
    - Posterior updates with observed data
    - Monte Carlo estimation of P(B > A)
    - Expected loss calculation for decision making
    - Plotting of posterior distributions
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Initialize with Beta prior hyperparameters.

        Args:
            prior_alpha: Alpha (pseudo-successes) for prior Beta distribution
            prior_beta: Beta (pseudo-failures) for prior Beta distribution
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.posteriors = {
            'A': [prior_alpha, prior_beta],
            'B': [prior_alpha, prior_beta]
        }

    def update(self, variant: str, successes: int, failures: int) -> None:
        """
        Update the posterior distribution with observed data.

        Conjugate update: alpha += successes, beta += failures

        Args:
            variant: 'A' or 'B'
            successes: Number of observed successes
            failures: Number of observed failures
        """
        if variant not in self.posteriors:
            raise ValueError(f"Unknown variant: {variant}. Use 'A' or 'B'.")
        self.posteriors[variant][0] += successes
        self.posteriors[variant][1] += failures

    def get_posterior(self, variant: str) -> tuple:
        """
        Get posterior parameters for a variant.

        Args:
            variant: 'A' or 'B'

        Returns:
            (alpha, beta) tuple of posterior parameters
        """
        return tuple(self.posteriors[variant])

    def _sample_beta(self, alpha: float, beta: float, n: int) -> np.ndarray:
        """
        Sample from Beta(alpha, beta) distribution.

        Args:
            alpha: Shape parameter alpha
            beta: Shape parameter beta
            n: Number of samples

        Returns:
            Array of n samples from Beta(alpha, beta)
        """
        return np.random.beta(alpha, beta, n)

    def probability_b_beats_a(self, n_simulations: int = 100000) -> float:
        """
        Estimate P(theta_B > theta_A) via Monte Carlo simulation.

        Samples from both posterior distributions and computes
        the fraction of samples where B exceeds A.

        Args:
            n_simulations: Number of Monte Carlo samples

        Returns:
            Estimated P(B > A)
        """
        alpha_a, beta_a = self.posteriors['A']
        alpha_b, beta_b = self.posteriors['B']
        samples_a = self._sample_beta(alpha_a, beta_a, n_simulations)
        samples_b = self._sample_beta(alpha_b, beta_b, n_simulations)
        return float(np.mean(samples_b > samples_a))

    def _beta_pdf(self, x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        Compute Beta distribution PDF from scratch using log-gamma (Lanczos approximation).

        log PDF(x; alpha, beta) = log_gamma(alpha+beta) - log_gamma(alpha) - log_gamma(beta)
                                  + (alpha-1)*log(x) + (beta-1)*log(1-x)

        Args:
            x: Array of values in (0, 1)
            alpha: Shape parameter alpha
            beta: Shape parameter beta

        Returns:
            Array of PDF values
        """
        log_norm = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
        log_pdf = (log_norm
                   + (alpha - 1) * np.log(np.maximum(x, 1e-300))
                   + (beta - 1) * np.log(np.maximum(1 - x, 1e-300)))
        return np.exp(log_pdf)

    def plot_posteriors(self, save_path: str = 'outputs/posteriors.png') -> None:
        """
        Plot the posterior Beta distributions for both variants.

        Saves a figure with histogram of posterior samples overlaid with
        the analytical PDF, and displays P(B > A).

        Args:
            save_path: Path to save the plot
        """
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        else:
            os.makedirs('outputs', exist_ok=True)

        x = np.linspace(0.001, 0.999, 1000)
        fig, ax = plt.subplots(figsize=(10, 6))

        for variant, color in [('A', 'steelblue'), ('B', 'coral')]:
            alpha, beta = self.posteriors[variant]
            samples = self._sample_beta(alpha, beta, 100000)
            ax.hist(samples, bins=100, density=True, alpha=0.4, color=color,
                    label=f'Variant {variant} (posterior samples)')
            pdf = self._beta_pdf(x, alpha, beta)
            ax.plot(x, pdf, color=color, lw=2,
                    label=f'Variant {variant}: Beta({alpha:.1f}, {beta:.1f})')

        ax.set_xlabel('Conversion Rate')
        ax.set_ylabel('Density')
        ax.set_title('Posterior Distributions for A/B Test Variants')
        ax.legend()

        prob = self.probability_b_beats_a()
        ax.text(0.7, 0.9, f'P(B > A) = {prob:.4f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Posteriors plot saved to {save_path}")

    def expected_loss(self, variant: str, n_simulations: int = 100000) -> float:
        """
        Compute the expected loss of choosing a given variant.

        E[loss(A)] = E[max(theta_B - theta_A, 0)]
        E[loss(B)] = E[max(theta_A - theta_B, 0)]

        A lower expected loss means the variant is preferable.

        Args:
            variant: 'A' or 'B'
            n_simulations: Number of Monte Carlo samples

        Returns:
            Expected loss value
        """
        alpha_a, beta_a = self.posteriors['A']
        alpha_b, beta_b = self.posteriors['B']
        samples_a = self._sample_beta(alpha_a, beta_a, n_simulations)
        samples_b = self._sample_beta(alpha_b, beta_b, n_simulations)

        if variant == 'A':
            return float(np.mean(np.maximum(samples_b - samples_a, 0)))
        elif variant == 'B':
            return float(np.mean(np.maximum(samples_a - samples_b, 0)))
        else:
            raise ValueError(f"Unknown variant: {variant}. Use 'A' or 'B'.")
