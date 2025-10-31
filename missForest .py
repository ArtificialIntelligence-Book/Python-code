"""
missForest Imputation Method Implementation in Python

missForest is an iterative imputation method that uses Random Forests
to predict missing values in datasets containing mixed data types 
(continuous and categorical). It iteratively imputes missing entries
by training random forest models on observed parts of the data.

Key steps:
- Initialize missing values (e.g., with column means/modes)
- Iterate over features with missing data:
    - For each feature, use it as the target and other features as predictors
    - Train a Random Forest on observed rows for that feature
    - Predict missing values for that feature
- Repeat iterations until convergence or max iterations reached

This implementation supports pandas DataFrames with mixed types.

Dependencies:
- numpy
- pandas
- scikit-learn

Install via:
pip install numpy pandas scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import check_random_state
from collections import defaultdict

class MissForestImputer:
    def __init__(self, max_iter=10, random_state=None, verbose=True):
        """
        Parameters:
        max_iter: int - max number of iterative imputations
        random_state: int or None - random seed for reproducibility
        verbose: bool - print progress info
        """
        self.max_iter = max_iter
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.imputed_df_ = None
        self.feature_types_ = None
        self.error_trace_ = []

    def _initial_imputation(self, X):
        """
        Initialize missing values:
        - For numeric columns: fill with column mean
        - For categorical columns: fill with column mode
        """
        X_init = X.copy()
        self.feature_types_ = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.feature_types_[col] = 'numeric'
                mean_val = X[col].mean()
                X_init[col] = X[col].fillna(mean_val)
            else:
                self.feature_types_[col] = 'categorical'
                mode_val = X[col].mode()
                mode_val = mode_val[0] if not mode_val.empty else np.nan
                X_init[col] = X[col].fillna(mode_val)
        return X_init

    def _fit_predict_feature(self, X_train, y_train, X_pred, feature_type):
        """
        Fit Random Forest on training data and predict on missing data
        """
        if feature_type == 'numeric':
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        return y_pred

    def fit_transform(self, X):
        """
        Perform iterative imputation on DataFrame X with missing values.

        Returns:
        imputed DataFrame with no missing values.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        X_imp = self._initial_imputation(X)
        if self.verbose:
            print("Initial imputation done.")

        # Identify columns with missing values
        missing_cols = X.columns[X.isnull().any()].tolist()
        if self.verbose:
            print(f"Columns with missing values: {missing_cols}")

        for iteration in range(self.max_iter):
            if self.verbose:
                print(f"\nIteration {iteration + 1}")

            X_old = X_imp.copy()

            for col in missing_cols:
                feature_type = self.feature_types_[col]

                # Rows where current feature is observed (training)
                observed_mask = ~X[col].isnull()
                # Rows where current feature is missing (to predict)
                missing_mask = X[col].isnull()

                if missing_mask.sum() == 0:
                    # No missing values in this column
                    continue

                # Features used for prediction (all other columns)
                X_train = X_imp.loc[observed_mask].drop(columns=[col])
                y_train = X_imp.loc[observed_mask, col]

                X_pred = X_imp.loc[missing_mask].drop(columns=[col])

                # Check if X_train or X_pred is empty or has invalid shape
                if X_train.shape[0] == 0 or X_pred.shape[0] == 0:
                    continue

                # Fit model and predict missing values
                y_pred = self._fit_predict_feature(X_train, y_train, X_pred, feature_type)

                # Assign predictions to imputed DataFrame
                X_imp.loc[missing_mask, col] = y_pred

            # Check convergence: stop if no change or minimal change
            diff = ((X_imp.select_dtypes(include=[np.number]) - X_old.select_dtypes(include=[np.number])).abs()).sum().sum()
            if self.verbose:
                print(f"Sum of absolute differences in numeric features: {diff:.6f}")
            self.error_trace_.append(diff)

            if diff < 1e-5:
                if self.verbose:
                    print("Convergence reached.")
                break

        self.imputed_df_ = X_imp
        return X_imp

# ==== Example Usage ====
if __name__ == "__main__":
    # Create example mixed data with missing values
    data = {
        'Age': [25, 30, np.nan, 35, 40, np.nan, 50],
        'Gender': ['M', 'F', 'F', np.nan, 'M', 'F', 'M'],
        'Income': [50000, 60000, 55000, np.nan, 65000, 70000, 62000],
        'Marital_Status': ['Single', 'Married', np.nan, 'Single', 'Married', 'Married', np.nan]
    }
    df = pd.DataFrame(data)

    print("Original Data with Missing Values:")
    print(df)

    # Apply missForest imputation
    imputer = MissForestImputer(max_iter=10, random_state=42, verbose=True)
    imputed_df = imputer.fit_transform(df)

    print("\nImputed Data:")
    print(imputed_df)