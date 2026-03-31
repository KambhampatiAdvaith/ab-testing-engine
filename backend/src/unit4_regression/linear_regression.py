import numpy as np


class MultipleLinearRegression:
    """
    Multiple Linear Regression using the Normal Equation.

    Model: y = X_b * beta + epsilon
    where X_b is X augmented with a bias column.

    Solution: beta = (X^T X)^{-1} X^T y

    Attributes:
        coefficients_: Fitted coefficients [intercept, beta_1, ..., beta_p]
        X_with_bias_: Design matrix with bias column
        y_: Target vector
        n_: Number of observations
        p_: Number of predictors (excluding intercept)
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the linear regression model using the Normal Equation.

        Solves: (X^T X) beta = X^T y

        Args:
            X: Feature matrix of shape (n, p)
            y: Target vector of shape (n,)
        """
        n = X.shape[0]
        ones = np.ones((n, 1))
        X_b = np.hstack([ones, X])

        XtX = X_b.T @ X_b
        Xty = X_b.T @ y
        self.coefficients_ = np.linalg.solve(XtX, Xty)

        self.X_with_bias_ = X_b
        self.y_ = y
        self.n_ = n
        self.p_ = X.shape[1]
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new data.

        Args:
            X: Feature matrix of shape (m, p)

        Returns:
            Predicted values of shape (m,)
        """
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack([ones, X])
        return X_b @ self.coefficients_

    def r_squared(self) -> float:
        """
        Compute the coefficient of determination R².

        R² = 1 - SS_res / SS_tot
        where SS_res = sum((y - y_hat)^2) and SS_tot = sum((y - y_mean)^2)

        Returns:
            R² value in [0, 1] (can be negative for very poor fits)
        """
        y_pred = self.X_with_bias_ @ self.coefficients_
        ss_res = float(np.sum((self.y_ - y_pred) ** 2))
        ss_tot = float(np.sum((self.y_ - np.mean(self.y_)) ** 2))
        if ss_tot == 0:
            return 1.0
        return float(1 - ss_res / ss_tot)

    def adjusted_r_squared(self) -> float:
        """
        Compute adjusted R² penalized for number of predictors.

        Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)

        Returns:
            Adjusted R² value
        """
        r2 = self.r_squared()
        n, p = self.n_, self.p_
        return float(1 - (1 - r2) * (n - 1) / (n - p - 1))

    def residuals(self) -> np.ndarray:
        """
        Compute residuals: e = y - y_hat.

        Returns:
            Residual vector of shape (n,)
        """
        return self.y_ - self.X_with_bias_ @ self.coefficients_

    def summary(self) -> None:
        """Print a formatted regression summary table with coefficient t-tests."""
        from .hypothesis_tests import t_test_coefficients
        t_results = t_test_coefficients(self)

        print("\n" + "=" * 70)
        print("REGRESSION SUMMARY")
        print("=" * 70)
        print(f"R²: {self.r_squared():.6f}   Adjusted R²: {self.adjusted_r_squared():.6f}")
        print(f"N observations: {self.n_}   N predictors: {self.p_}")
        print("-" * 70)
        print(f"{'Coefficient':<20} {'Estimate':>12} {'Std Error':>12} {'t-stat':>10} {'p-value':>10}")
        print("-" * 70)
        names = ['Intercept'] + [f'X{i}' for i in range(1, self.p_ + 1)]
        for i, name in enumerate(names):
            coef = self.coefficients_[i]
            se = t_results['std_errors'][i]
            t = t_results['t_statistics'][i]
            p = t_results['p_values'][i]
            sig = '*' if p < 0.05 else ''
            print(f"{name:<20} {coef:>12.6f} {se:>12.6f} {t:>10.4f} {p:>10.4f} {sig}")
        print("=" * 70)
