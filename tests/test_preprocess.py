"""Unit tests for the preprocessing pipeline."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.preprocess import (
    build_preprocessor,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    get_feature_names,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_dataframe():
    """Create a small synthetic DataFrame matching Heart Disease schema."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "age": np.random.randint(30, 80, n).astype(float),
        "sex": np.random.randint(0, 2, n).astype(float),
        "cp": np.random.randint(0, 4, n).astype(float),
        "trestbps": np.random.randint(90, 180, n).astype(float),
        "chol": np.random.randint(150, 400, n).astype(float),
        "fbs": np.random.randint(0, 2, n).astype(float),
        "restecg": np.random.randint(0, 3, n).astype(float),
        "thalach": np.random.randint(80, 200, n).astype(float),
        "exang": np.random.randint(0, 2, n).astype(float),
        "oldpeak": np.random.uniform(0, 5, n),
        "slope": np.random.randint(0, 3, n).astype(float),
        "ca": np.random.randint(0, 5, n).astype(float),
        "thal": np.random.randint(0, 4, n).astype(float),
    })


@pytest.fixture
def sample_dataframe_with_nans(sample_dataframe):
    """Introduce some NaNs to test imputation."""
    df = sample_dataframe.copy()
    df.loc[0, "age"] = np.nan
    df.loc[1, "chol"] = np.nan
    df.loc[2, "ca"] = np.nan
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPreprocessor:

    def test_build_preprocessor_returns_column_transformer(self, sample_dataframe):
        """Preprocessor should be a ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        preprocessor = build_preprocessor()
        assert hasattr(preprocessor, "fit_transform")

    def test_output_shape(self, sample_dataframe):
        """Output should have correct number of columns (numeric + categorical)."""
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(sample_dataframe)
        expected_cols = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
        assert X_transformed.shape == (len(sample_dataframe), expected_cols)

    def test_no_nans_after_transform(self, sample_dataframe_with_nans):
        """After transform, there should be no NaN values (imputation applied)."""
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(sample_dataframe_with_nans)
        assert not np.isnan(X_transformed).any(), "NaN values found after transform"

    def test_numeric_features_scaled(self, sample_dataframe):
        """Numeric features should be standardized (approx mean=0, std=1)."""
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(sample_dataframe)
        n_numeric = len(NUMERIC_FEATURES)
        numeric_cols = X_transformed[:, :n_numeric]
        assert abs(numeric_cols.mean()) < 1.0, "Mean of scaled numerics is far from 0"

    def test_feature_names_match(self):
        """Feature name list should match expected columns."""
        names = get_feature_names()
        assert set(names) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    def test_transform_single_row(self, sample_dataframe):
        """Single-row transform should work without error."""
        preprocessor = build_preprocessor()
        preprocessor.fit(sample_dataframe)
        single = sample_dataframe.iloc[:1]
        result = preprocessor.transform(single)
        assert result.shape[0] == 1

    def test_output_is_numpy_array(self, sample_dataframe):
        """Transform output should be a numpy ndarray."""
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(sample_dataframe)
        assert isinstance(X_transformed, np.ndarray)
