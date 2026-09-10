import numpy as np
import pandas as pd


class FeatureOrthogonalizer:
    """
    Transforms feature matrix via Principal Component Decorrelation
    to guarantee orthogonal representations across all 18 indicators.
    """
    def __init__(self, variance_explained: float = 0.95):
        self.variance_explained = variance_explained
        self.mean = None
        self.std = None
        self.components = None

    def fit_transform(self, df: pd.DataFrame, exclude_cols: list = ["close"]) -> pd.DataFrame:
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].values

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.mean) / self.std

        cov = np.cov(X_norm, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvalues descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select variance threshold
        cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        num_components = np.argmax(cum_var >= self.variance_explained) + 1

        self.components = eigenvectors[:, :num_components]
        X_ortho = np.dot(X_norm, self.components)

        ortho_cols = [f"pc_{i+1}" for i in range(num_components)]
        out_df = pd.DataFrame(X_ortho, columns=ortho_cols, index=df.index)

        for col in exclude_cols:
            if col in df.columns:
                out_df[col] = df[col]

        return out_df
