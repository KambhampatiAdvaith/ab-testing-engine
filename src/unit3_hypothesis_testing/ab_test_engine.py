import numpy as np
import os

# Named constants for binary search used in z/power critical value lookups
_Z_SEARCH_UPPER_BOUND = 10.0    # Upper bound for z-value binary search
_BINARY_SEARCH_ITERATIONS = 100  # Number of binary search iterations for convergence


def _erf(z: float) -> float:
    """Abramowitz and Stegun approximation of erf, max error 1.5e-7."""
    if z == 0:
        return 0.0
    t = 1.0 / (1.0 + 0.3275911 * abs(z))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    result = 1.0 - poly * np.exp(-z * z)
    return float(np.sign(z)) * result


class ABTestEngine:
    """
    Engine for running A/B tests using frequentist statistical methods.

    All statistical computations are implemented from scratch using NumPy only.
    Supports:
    - Two-proportion z-test
    - Confidence interval calculation
    - Sample size calculation
    - Full experiment reporting
    """

    def _normal_cdf(self, x: float) -> float:
        """
        Standard normal CDF using error function approximation.

        CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))

        Args:
            x: Value at which to evaluate the CDF

        Returns:
            P(Z <= x) for Z ~ N(0,1)
        """
        return 0.5 * (1.0 + _erf(x / np.sqrt(2)))

    def _z_critical(self, alpha: float) -> float:
        """
        Get the z critical value for a given significance level (two-tailed).

        Finds z such that P(Z <= z) = 1 - alpha/2 using binary search.

        Args:
            alpha: Significance level (e.g., 0.05 for 95% CI)

        Returns:
            z critical value
        """
        target = 1 - alpha / 2
        lo, hi = 0.0, _Z_SEARCH_UPPER_BOUND
        for _ in range(_BINARY_SEARCH_ITERATIONS):
            mid = (lo + hi) / 2
            if self._normal_cdf(mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def run_ztest(self, control_clicks: int, control_total: int,
                  variant_clicks: int, variant_total: int,
                  alpha: float = 0.05) -> dict:
        """
        Two-proportion z-test for comparing conversion rates.

        Math:
            p1 = control_clicks / control_total
            p2 = variant_clicks / variant_total
            p_pooled = (control_clicks + variant_clicks) / (control_total + variant_total)
            se_pooled = sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            z = (p2 - p1) / se_pooled
            p_value = 2 * (1 - normal_cdf(|z|))
            CI: (p2 - p1) ± z_alpha/2 * se_unpooled

        Args:
            control_clicks: Number of conversions in control group
            control_total: Total users in control group
            variant_clicks: Number of conversions in variant group
            variant_total: Total users in variant group
            alpha: Significance level (default 0.05)

        Returns:
            Dict with z_statistic, p_value, confidence_interval, is_significant,
            control_rate, variant_rate, relative_uplift, conclusion
        """
        p1 = control_clicks / control_total
        p2 = variant_clicks / variant_total
        n1 = control_total
        n2 = variant_total

        p_pooled = (control_clicks + variant_clicks) / (control_total + variant_total)
        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

        if se_pooled == 0:
            z_stat = 0.0
        else:
            z_stat = float((p2 - p1) / se_pooled)

        p_value = float(2 * (1 - self._normal_cdf(abs(z_stat))))

        # Confidence interval using unpooled SE
        se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z_crit = self._z_critical(alpha)
        diff = p2 - p1
        ci_lower = float(diff - z_crit * se_unpooled)
        ci_upper = float(diff + z_crit * se_unpooled)

        is_significant = bool(p_value < alpha)

        if is_significant:
            direction = "higher" if p2 > p1 else "lower"
            conclusion = (
                f"Variant conversion rate ({p2:.4f}) is statistically significantly "
                f"{direction} than control ({p1:.4f}) at alpha={alpha}."
            )
        else:
            conclusion = (
                f"No statistically significant difference detected between "
                f"control ({p1:.4f}) and variant ({p2:.4f}) at alpha={alpha}."
            )

        return {
            'z_statistic': float(z_stat),
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': is_significant,
            'control_rate': float(p1),
            'variant_rate': float(p2),
            'relative_uplift': float((p2 - p1) / p1) if p1 > 0 else 0.0,
            'conclusion': conclusion
        }

    def calculate_sample_size(self, baseline_rate: float, min_detectable_effect: float,
                               alpha: float = 0.05, power: float = 0.8) -> int:
        """
        Calculate the required sample size per group for a two-proportion z-test.

        Formula:
            n = (z_alpha/2 + z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p2 - p1)^2

        Args:
            baseline_rate: Expected conversion rate in the control group
            min_detectable_effect: Minimum effect size to detect
            alpha: Significance level (default 0.05)
            power: Statistical power (default 0.80)

        Returns:
            Required sample size per group (integer)
        """
        p1 = baseline_rate
        p2 = baseline_rate + min_detectable_effect

        z_alpha = self._z_critical(alpha)

        # Find z_beta: CDF(z_beta) = power
        lo, hi = 0.0, _Z_SEARCH_UPPER_BOUND
        for _ in range(_BINARY_SEARCH_ITERATIONS):
            mid = (lo + hi) / 2
            if self._normal_cdf(mid) < power:
                lo = mid
            else:
                hi = mid
        z_beta = (lo + hi) / 2

        numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
        denominator = (p2 - p1) ** 2

        return int(np.ceil(numerator / denominator))

    def run_experiment(self, control_data: dict, variant_data: dict) -> dict:
        """
        Run a full A/B experiment and produce a comprehensive report.

        Args:
            control_data: Dict with 'clicks' (int) and 'total' (int) keys
            variant_data: Dict with 'clicks' (int) and 'total' (int) keys

        Returns:
            Dict combining z-test results with data and sample size recommendation
        """
        result = self.run_ztest(
            control_clicks=control_data['clicks'],
            control_total=control_data['total'],
            variant_clicks=variant_data['clicks'],
            variant_total=variant_data['total']
        )
        result['control_data'] = control_data
        result['variant_data'] = variant_data
        mde = abs(result['variant_rate'] - result['control_rate']) or 0.01
        result['sample_size_recommendation'] = self.calculate_sample_size(
            baseline_rate=result['control_rate'],
            min_detectable_effect=mde
        )
        return result
