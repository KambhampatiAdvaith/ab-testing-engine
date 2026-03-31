import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def demonstrate_clt(population_distribution, sample_sizes: list,
                    n_simulations: int = 1000) -> None:
    """
    Demonstrate the Central Limit Theorem visually.

    For each sample size n in sample_sizes:
    - Draw n_simulations samples of that size from population_distribution
    - Compute the mean of each sample
    - Plot the distribution of sample means
    - Overlay the theoretical normal curve N(mu, sigma^2/n)

    As n grows, the sampling distribution converges to the normal distribution
    regardless of the underlying population distribution (CLT).

    Saves figure to outputs/clt_demonstration.png

    Args:
        population_distribution: Distribution object with .sample(), .mean(), .variance() methods
        sample_sizes: List of sample sizes to demonstrate
        n_simulations: Number of samples to draw per sample size
    """
    os.makedirs('outputs', exist_ok=True)

    n_plots = len(sample_sizes)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    pop_mean = population_distribution.mean()
    pop_var = population_distribution.variance()

    for ax, n in zip(axes, sample_sizes):
        sample_means = np.array([
            np.mean(population_distribution.sample(n))
            for _ in range(n_simulations)
        ])

        ax.hist(sample_means, bins=40, density=True, alpha=0.7,
                color='steelblue', label='Sample means')

        # Theoretical normal curve via CLT
        theoretical_std = np.sqrt(pop_var / n)
        x = np.linspace(sample_means.min(), sample_means.max(), 200)
        pdf_vals = (1.0 / (theoretical_std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - pop_mean) / theoretical_std) ** 2
        )
        ax.plot(x, pdf_vals, 'r-', lw=2,
                label=f'N({pop_mean:.2f}, {theoretical_std ** 2:.4f})')

        ax.set_title(f'n = {n}')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    fig.suptitle('Central Limit Theorem Demonstration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/clt_demonstration.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("CLT demonstration saved to outputs/clt_demonstration.png")
