"""
Air Pollution Forecasting with Multivariate LSTM & MLOps Pipeline
IEEE Research Implementation

Title: "Multivariate Time Series Analysis and Batch Normalization for Air Quality Prediction in Long Short-Term Memory Networks"
DOI: 10.1109/INOCON60754.2024.10511808
Conference: 2024 3rd International Conference for Innovation in Technology (INOCON)

Author: Tirumala Manav
GitHub: https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps
Email: Contact via GitHub Issues
Date: July 2024

Description: [Add specific module description here]

License: MIT License
Copyright (c) 2024 Tirumala Manav

This module implements [specific functionality] as part of the complete
air pollution forecasting system with production-ready MLOps pipeline.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

# class CategoricalImputer(BaseEstimator, TransformerMixin):
#     def __init__(self, variables=None) -> None:
#         if variables is None:
#             variables = []
#         if not isinstance(variables, list):
#             variables = [variables]
#         self.variables = variables
#         print(f"Initialized CategoricalImputer with variables: {self.variables}")

#     def fit(self, X, y=None):
#         self.imputer_dict_ = {}
#         if isinstance(X, pd.DataFrame):
#             print("Fitting CategoricalImputer with DataFrame")
#             for feature in self.variables:
#                 if feature in X.columns:
#                     self.imputer_dict_[feature] = X[feature].mode()[0]
#                     print(f"Imputed value for {feature}: {self.imputer_dict_[feature]}")
#                 else:
#                     print(f"Feature {feature} not in DataFrame columns")
#         else:  # X is a numpy array
#             print("Fitting CategoricalImputer with NumPy array")
#             X_df = pd.DataFrame(X, columns=self.variables)
#             print(f"Created DataFrame from NumPy array with columns: {self.variables}")
#             for feature in self.variables:
#                 if feature in X_df.columns:
#                     self.imputer_dict_[feature] = X_df[feature].mode()[0]
#                     print(f"Imputed value for {feature}: {self.imputer_dict_[feature]}")
#                 else:
#                     print(f"Feature {feature} not in DataFrame columns")
#         return self

#     def transform(self, X):
#         if isinstance(X, pd.DataFrame):
#             print("Transforming DataFrame with CategoricalImputer")
#             X = X.copy()
#             for feature in self.variables:
#                 if feature in X.columns:
#                     X[feature].fillna(self.imputer_dict_[feature], inplace=True)
#                     print(f"Filled NaN in {feature} with {self.imputer_dict_[feature]}")
#                 else:
#                     print(f"Feature {feature} not in DataFrame columns")
#         else:  # X is a numpy array
#             print("Transforming NumPy array with CategoricalImputer")
#             X_df = pd.DataFrame(X, columns=self.variables)
#             print(f"Created DataFrame from NumPy array with columns: {self.variables}")
#             for feature in self.variables:
#                 if feature in X_df.columns:
#                     X_df[feature].fillna(self.imputer_dict_[feature], inplace=True)
#                     print(f"Filled NaN in {feature} with {self.imputer_dict_[feature]}")
#                 else:
#                     print(f"Feature {feature} not in DataFrame columns")
#             X = X_df.values
#         return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None) -> None:
        if variables is None:
            variables = []
        if not isinstance(variables, list):
            variables = [variables]
        self.variables = variables
        print(f"Initialized NumericalImputer with variables: {self.variables}")

    def fit(self, X, y=None):
        self.imputer_dict_ = {}
        if isinstance(X, pd.DataFrame):
            print("Fitting NumericalImputer with DataFrame")
            for feature in self.variables:
                if feature in X.columns:
                    self.imputer_dict_[feature] = X[feature].mean()
                    print(f"Imputed mean value for {feature}: {self.imputer_dict_[feature]}")
                else:
                    print(f"Feature {feature} not in DataFrame columns")
        else:  # X is a numpy array
            print("Fitting NumericalImputer with NumPy array")
            X_df = pd.DataFrame(X, columns=self.variables)
            for feature in self.variables:
                if feature in X_df.columns:
                    self.imputer_dict_[feature] = X_df[feature].mean()
                    print(f"Imputed mean value for {feature}: {self.imputer_dict_[feature]}")
                else:
                    print(f"Feature {feature} not in DataFrame columns")
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            print("Transforming DataFrame with NumericalImputer")
            X = X.copy()
            for feature in self.variables:
                if feature in X.columns:
                    X[feature].fillna(self.imputer_dict_[feature], inplace=True)
                    print(f"Filled NaN in {feature} with {self.imputer_dict_[feature]}")
                else:
                    print(f"Feature {feature} not in DataFrame columns")
        else:  # X is a numpy array
            print("Transforming NumPy array with NumericalImputer")
            X_df = pd.DataFrame(X, columns=self.variables)
            for feature in self.variables:
                if feature in X_df.columns:
                    X_df[feature].fillna(self.imputer_dict_[feature], inplace=True)
                    print(f"Filled NaN in {feature} with {self.imputer_dict_[feature]}")
                else:
                    print(f"Feature {feature} not in DataFrame columns")
            X = X_df.values
        return X

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None, reference_variable=None) -> None:
        if variables_to_drop is None:
            variables_to_drop = []
        if not isinstance(variables_to_drop, list):
            variables_to_drop = [variables_to_drop]

        self.variables = variables_to_drop
        if reference_variable in self.variables:
            self.variables.remove(reference_variable)
        print(f"Initialized DropColumns with variables: {self.variables}")

    def fit(self, X, y=None):
        print("Fitting DropColumns")
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            print("Transforming DataFrame with DropColumns")
            X = X.copy()
            print(f"DataFrame shape before dropping columns: {X.shape}")
            X.drop(columns=self.variables, inplace=True, errors='ignore')
            print(f"DataFrame shape after dropping columns: {X.shape}")
        else:  # X is a numpy array
            print("Transforming NumPy array with DropColumns")
            X_df = pd.DataFrame(X)
            print(f"Array shape before dropping columns: {X_df.shape}")
            X_df.drop(columns=self.variables, inplace=True, errors='ignore')
            X = X_df.values
            print(f"Array shape after dropping columns: {X.shape}")
        return X

# class CategoricalEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, variables=None):
#         if variables is None:
#             variables = []
#         if not isinstance(variables, list):
#             variables = [variables]
#         self.variables = variables
#         self.encoder = OneHotEncoder(sparse=False)
#         print(f"Initialized CategoricalEncoder with variables: {self.variables}")

#     def fit(self, X, y=None):
#         if isinstance(X, pd.DataFrame):
#             print("Fitting CategoricalEncoder with DataFrame")
#             self.encoder.fit(X[self.variables])
#         else:  # X is a numpy array
#             print("Fitting CategoricalEncoder with NumPy array")
#             X_df = pd.DataFrame(X, columns=self.variables)
#             self.encoder.fit(X_df[self.variables])
#         return self

#     def transform(self, X):
#         if isinstance(X, pd.DataFrame):
#             print("Transforming DataFrame with CategoricalEncoder")
#             encoded_columns = self.encoder.transform(X[self.variables])
#             print(f"Encoded columns shape: {encoded_columns.shape}")
#             X = X.drop(columns=self.variables)
#             X = pd.concat([X, pd.DataFrame(encoded_columns, index=X.index)], axis=1)
#         else:  # X is a numpy array
#             print("Transforming NumPy array with CategoricalEncoder")
#             X_df = pd.DataFrame(X)
#             encoded_columns = self.encoder.transform(X_df[self.variables])
#             print(f"Encoded columns shape: {encoded_columns.shape}")
#             X_df = X_df.drop(columns=self.variables)
#             X = np.hstack([X_df.values, encoded_columns])
#         return X

class SomePreprocessingClass(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables if variables is not None else []
        print(f"Initialized SomePreprocessingClass with variables: {self.variables}")
        self.numerical_imputer = NumericalImputer(variables=self.variables)
        # self.categorical_imputer = None  # No categorical imputer needed

    def fit(self, X: np.ndarray, y=None):
        print(f"Original X shape for fit: {X.shape}")
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X

        print(f"Reshaped X shape for fit: {X_reshaped.shape}")

        if not self.variables:
            self.variables = [f'feature_{i}' for i in range(X_reshaped.shape[1])]
            print(f"Variables set to default: {self.variables}")

        print(f"Variables for DataFrame: {self.variables}")

        X_df = pd.DataFrame(X_reshaped, columns=self.variables)
        print(f"DataFrame shape for fit: {X_df.shape}")

        self.numerical_imputer.fit(X_df)
        return self

    def transform(self, X: np.ndarray):
        print(f"Original X shape for transform: {X.shape}")
        if len(X.shape) == 3:
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            X_reshaped = X

        print(f"Reshaped X shape for transform: {X_reshaped.shape}")

        if not self.variables:
            self.variables = [f'feature_{i}' for i in range(X_reshaped.shape[1])]
            print(f"Variables set to default: {self.variables}")

        print(f"Variables for DataFrame: {self.variables}")

        X_df = pd.DataFrame(X_reshaped, columns=self.variables)
        print(f"DataFrame shape for transform: {X_df.shape}")

        X_imputed = self.numerical_imputer.transform(X_df)
        print(f"Imputed DataFrame shape: {X_imputed.shape}")

        X_imputed_array = np.array(X_imputed)

        print(f"NumPy array shape after imputation: {X_imputed_array.shape}")

        print(f"Final transformed shape: {X_imputed_array.shape}")

        return X_imputed_array
