"""
Advanced A/B Testing & User Segmentation Engine - CLI Entry Point

Usage:
    python main.py --unit 1   # CLT & distributions demo
    python main.py --unit 3   # A/B testing demo
    python main.py --unit 4   # Regression & LTV demo
    python main.py --unit 5   # Clustering & personas demo
    python main.py --all      # Run all demos
"""
import argparse
import numpy as np
import os

os.makedirs('outputs', exist_ok=True)


def run_unit1() -> None:
    """CLT visualization and distributions demo."""
    print("\n" + "=" * 60)
    print("UNIT 1-2: Random Variables & CLT Demonstration")
    print("=" * 60)

    from src.unit1_2_random_variables.distributions import (
        BinomialDistribution, NormalDistribution, UniformDistribution
    )
    from src.unit1_2_random_variables.traffic_simulator import simulate_traffic
    from src.unit1_2_random_variables.clt_visualizer import demonstrate_clt

    binom = BinomialDistribution(n=10, p=0.3)
    print(f"\nBinomial(n=10, p=0.3):")
    print(f"  PMF(k=3) = {binom.pmf(3):.6f}")
    print(f"  Mean = {binom.mean():.2f}, Variance = {binom.variance():.2f}")

    norm = NormalDistribution(mu=0, sigma=1)
    print(f"\nNormal(mu=0, sigma=1):")
    print(f"  PDF(0) = {norm.pdf(0):.6f}")
    print(f"  CDF(0) = {norm.cdf(0):.6f}")

    uniform = UniformDistribution(a=0, b=1)
    print(f"\nRunning CLT demonstration with Uniform(0,1) population...")
    demonstrate_clt(uniform, sample_sizes=[5, 30, 100], n_simulations=1000)

    print("\nSimulating website traffic (100 users, 7 days)...")
    traffic = simulate_traffic(n_users=100, n_days=7)
    print(f"  Total records: {len(traffic['user_id'])}")
    print(f"  Avg clicks: {traffic['clicks'].mean():.3f}")
    print(f"  Avg session time: {traffic['session_time'].mean():.2f} min")


def run_unit3() -> None:
    """A/B testing and Bayesian estimation demo."""
    print("\n" + "=" * 60)
    print("UNIT 3: A/B Testing & Bayesian Estimation")
    print("=" * 60)

    from src.unit3_hypothesis_testing.ab_test_engine import ABTestEngine
    from src.unit3_hypothesis_testing.bayesian_estimation import BayesianABTest
    from src.utils.data_generator import generate_conversion_data

    engine = ABTestEngine()
    control, variant = generate_conversion_data(
        n_control=1000, n_variant=1000,
        control_rate=0.10, variant_rate=0.13
    )

    print(f"\nControl: {control['clicks']}/{control['total']} conversions")
    print(f"Variant: {variant['clicks']}/{variant['total']} conversions")

    result = engine.run_ztest(
        control['clicks'], control['total'],
        variant['clicks'], variant['total']
    )

    print(f"\nZ-Test Results:")
    print(f"  Z-statistic: {result['z_statistic']:.4f}")
    print(f"  P-value: {result['p_value']:.6f}")
    print(f"  95% CI for diff: ({result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f})")
    print(f"  Significant: {result['is_significant']}")
    print(f"  {result['conclusion']}")

    n_needed = engine.calculate_sample_size(baseline_rate=0.10, min_detectable_effect=0.02)
    print(f"\nRequired sample size (per group) for 2% MDE: {n_needed:,}")

    print("\nBayesian A/B Test:")
    bayes = BayesianABTest(prior_alpha=1, prior_beta=1)
    bayes.update('A', successes=control['clicks'], failures=control['total'] - control['clicks'])
    bayes.update('B', successes=variant['clicks'], failures=variant['total'] - variant['clicks'])

    prob = bayes.probability_b_beats_a()
    loss_a = bayes.expected_loss('A')
    loss_b = bayes.expected_loss('B')

    print(f"  P(B > A) = {prob:.4f}")
    print(f"  Expected loss(A) = {loss_a:.6f}")
    print(f"  Expected loss(B) = {loss_b:.6f}")

    bayes.plot_posteriors('outputs/posteriors.png')


def run_unit4() -> None:
    """Multiple linear regression and LTV analysis demo."""
    print("\n" + "=" * 60)
    print("UNIT 4: Multiple Linear Regression")
    print("=" * 60)

    from src.unit4_regression.linear_regression import MultipleLinearRegression
    from src.unit4_regression.hypothesis_tests import f_test_overall
    from src.unit4_regression.lifetime_value import generate_ltv_dataset, analyze_feature_importance

    X, y, feature_names = generate_ltv_dataset(n_users=500)

    model = MultipleLinearRegression()
    model.fit(X, y)

    print(f"\nLTV Regression Results:")
    print(f"  R²: {model.r_squared():.6f}")
    print(f"  Adjusted R²: {model.adjusted_r_squared():.6f}")

    f_result = f_test_overall(model)
    print(f"  F-statistic: {f_result['f_statistic']:.4f}")
    print(f"  F-test p-value: {f_result['p_value']:.6f}")

    model.summary()
    analyze_feature_importance(model, feature_names)


def run_unit5() -> None:
    """K-Means clustering and user persona discovery demo."""
    print("\n" + "=" * 60)
    print("UNIT 5: K-Means Clustering & User Personas")
    print("=" * 60)

    from src.unit5_clustering.kmeans import KMeans
    from src.unit5_clustering.user_personas import (
        generate_user_behavior_data, discover_personas,
        analyze_personas, visualize_personas
    )

    data = generate_user_behavior_data(n_users=400)
    print(f"\nGenerated {data.shape[0]} users with {data.shape[1]} features")

    km = KMeans(k=2)
    inertias = km.elbow_method(data, k_range=range(2, 8))
    print(f"Inertias for k=2..7: {[f'{v:.0f}' for v in inertias]}")

    labels = discover_personas(data, k=4)
    personas = analyze_personas(data, labels)

    print("\nDiscovered User Personas:")
    for cid, info in personas.items():
        print(f"  Cluster {cid} ({info['label']}): {info['size']} users")
        print(f"    Avg pages/session: {info['means']['pages_per_session']:.1f}")
        print(f"    Avg session duration: {info['means']['avg_session_duration']:.1f} min")

    visualize_personas(data, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced A/B Testing & User Segmentation Engine')
    parser.add_argument('--unit', type=int, choices=[1, 3, 4, 5],
                        help='Run demo for specific unit (1, 3, 4, or 5)')
    parser.add_argument('--all', action='store_true', help='Run all demos')

    args = parser.parse_args()

    if args.all:
        run_unit1()
        run_unit3()
        run_unit4()
        run_unit5()
    elif args.unit == 1:
        run_unit1()
    elif args.unit == 3:
        run_unit3()
    elif args.unit == 4:
        run_unit4()
    elif args.unit == 5:
        run_unit5()
    else:
        parser.print_help()
